"""
ZED Mini IMU reader — projected_gravity + ang_vel_z via Madgwick filter.

Does NOT call enable_positional_tracking() — raw IMU only (no ZED depth needed
for the GRU+ProprioTorque policy).

Dependencies:
  pip install imufusion
  ZED SDK 5.2.2 (already on Jetson)

Standalone test:
  python imu_reader.py
  → prints projected_gravity and ang_vel_z at ~10 Hz for 5 seconds.
  When robot is flat and upright, projected_gravity should be ≈ [0, 0, -1].
"""

import math
import time

import numpy as np

try:
    import pyzed.sl as sl
except ImportError:
    raise ImportError("pyzed not found — install ZED SDK 5.2.2 on Jetson")

try:
    import imufusion
except ImportError:
    raise ImportError("imufusion not found — run: pip install imufusion")


class ImuReader:
    """
    Reads ZED Mini IMU at ~200 Hz internally, runs Madgwick AHRS filter,
    and exposes projected_gravity + ang_vel_z at whatever rate read() is called.
    """

    def __init__(self, sample_rate: int = 200):
        self._zed = sl.Camera()

        init = sl.InitParameters()
        init.sdk_verbose = 0
        status = self._zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED Mini open failed: {status}")

        # Raw IMU only — do NOT call enable_positional_tracking
        self._sensors_data = sl.SensorsData()

        # Madgwick AHRS — NWU convention matches Isaac Lab body frame
        self._ahrs = imufusion.Ahrs()
        settings = imufusion.Settings()
        settings.convention = imufusion.CONVENTION_NWU
        settings.gain = 0.5
        settings.gyroscope_range = 2000
        settings.acceleration_rejection = 10.0
        settings.magnetic_rejection = 10.0
        settings.recovery_trigger_period = 5 * sample_rate
        self._ahrs.settings = settings
        self._dt = 1.0 / sample_rate

        # Warm up filter for ~0.5 s before returning valid readings
        print("Warming up IMU filter...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.5:
            self._zed.get_sensors_data(self._sensors_data, sl.TIME_REFERENCE.CURRENT)
            imu = self._sensors_data.get_imu_data()
            gyr = imu.get_angular_velocity()
            acc = imu.get_linear_acceleration()
            self._ahrs.update_no_magnetometer(
                np.array([gyr.x, gyr.y, gyr.z]),
                np.array([acc.x, acc.y, acc.z]),
                self._dt,
            )
            time.sleep(self._dt)
        print("IMU ready.")

    def read(self) -> tuple[np.ndarray, float, float]:
        """
        Returns:
            projected_gravity: np.ndarray (3,) — gravity vector in body frame.
                               ≈ [0, 0, -1] when robot is flat and upright.
            ang_vel_z: float — yaw rate in rad/s (positive = left turn).
            acc_world_x: float — forward linear acceleration in world frame (m/s²),
                         gravity removed. Used by ObsBuilder for slip estimation.
        """
        self._zed.get_sensors_data(self._sensors_data, sl.TIME_REFERENCE.CURRENT)
        imu = self._sensors_data.get_imu_data()
        gyr = imu.get_angular_velocity()    # deg/s
        acc = imu.get_linear_acceleration() # m/s²

        ax, ay, az = float(acc.x), float(acc.y), float(acc.z)

        self._ahrs.update_no_magnetometer(
            np.array([gyr.x, gyr.y, gyr.z]),
            np.array([ax, ay, az]),
            self._dt,
        )

        # Quaternion (w, x, y, z) — body-to-world rotation
        q = self._ahrs.quaternion.wxyz
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])

        # Rotate world gravity [0, 0, -1] into body frame: R^T @ [0,0,-1]
        gx = -2.0 * (x*z - w*y)
        gy = -2.0 * (y*z + w*x)
        gz = -(w*w - x*x - y*y + z*z)
        projected_gravity = np.array([gx, gy, gz], dtype=np.float32)

        ang_vel_z = float(math.radians(gyr.z))  # deg/s → rad/s

        # Rotate body-frame acceleration into world frame, then remove gravity.
        # World x = forward direction. R(q) @ a_body gives world-frame acceleration.
        # Row 0 of rotation matrix from quaternion:
        acc_world_x = (ax * (w*w + x*x - y*y - z*z)
                       + ay * 2.0 * (x*y - w*z)
                       + az * 2.0 * (x*z + w*y))
        # Gravity in world frame is [0, 0, -9.81]; world-x component is 0 on flat
        # ground and non-zero on slopes — no explicit removal needed for slip detection.

        return projected_gravity, ang_vel_z, acc_world_x

    def close(self) -> None:
        self._zed.close()


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    reader = ImuReader()
    print("\nReading for 5 s — robot should be flat. projected_gravity ≈ [0, 0, -1]")
    print(f"{'proj_grav_x':>12} {'proj_grav_y':>12} {'proj_grav_z':>12} {'ang_vel_z':>12} {'acc_world_x':>12}")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5.0:
        grav, wz, ax = reader.read()
        print(f"{grav[0]:12.4f} {grav[1]:12.4f} {grav[2]:12.4f} {wz:12.4f} {ax:12.4f}")
        time.sleep(0.1)
    reader.close()
