"""
ZED Mini → terrain_scan[35] pipeline for AWM real-world deployment.

Replicates the sim ray caster: 7x5 grid at 0.15m resolution,
covering 0.9m wide x 0.6m deep ahead of the robot.

Output: terrain_scan[35] — relative heights in robot body frame,
        matching the observation the policy was trained on.

Standalone test:
    python3 zed_terrain_scan.py

Import usage:
    from zed_terrain_scan import TerrainScanner
    scanner = TerrainScanner()
    scanner.start()
    scan = scanner.get_terrain_scan(robot_base_z=0.0)
    gravity, yaw_rate = scanner.get_imu_data()
    scanner.stop()
"""

import numpy as np
import pyzed.sl as sl


# ---------------------------------------------------------------------------
# Grid parameters — must match sim ray caster exactly
# ---------------------------------------------------------------------------
GRID_NX = 7           # columns along forward (x) axis
GRID_NY = 5           # rows along lateral (y) axis
GRID_RES = 0.15       # metres between grid points
GRID_X_START = 0.15   # distance of first column ahead (metres)
GRID_Y_START = -0.30  # lateral offset of first row (metres, negative = right)
SEARCH_RADIUS = 0.07  # search radius per grid cell (half of GRID_RES)

# Pre-computed grid positions in robot body frame
GRID_X = np.array([GRID_X_START + i * GRID_RES for i in range(GRID_NX)])  # [0.15 .. 1.05]
GRID_Y = np.array([GRID_Y_START + j * GRID_RES for j in range(GRID_NY)])  # [-0.30 .. 0.30]


class TerrainScanner:
    """
    Wraps ZED Mini SDK 5.x to produce terrain_scan[35] and IMU data.

    Camera mounting parameters (adjust to match your physical setup):
        cam_height:   metres above robot base z
        cam_forward:  metres forward of robot base origin
        tilt_deg:     downward tilt angle in degrees
        yaw_alpha:    low-pass filter coefficient for yaw_rate (0=frozen, 1=raw).
                      Default 0.2 smooths gyro flicker without adding much lag.
    """

    def __init__(
        self,
        cam_height: float = 0.15,
        cam_forward: float = 0.05,
        tilt_deg: float = 30.0,
        yaw_alpha: float = 0.2,
        resolution=sl.RESOLUTION.HD720,
        depth_mode=sl.DEPTH_MODE.ULTRA,
    ):
        self._camera = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = resolution
        init_params.depth_mode = depth_mode
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
        init_params.depth_minimum_distance = 0.1  # ZED Mini supports 0.1m (vs 0.3m for ZED 2i)
        init_params.depth_maximum_distance = 2.0
        self._init_params = init_params

        self._runtime = sl.RuntimeParameters()
        self._runtime.confidence_threshold = 80

        self._point_cloud = sl.Mat()
        self._sensors_data = sl.SensorsData()

        # Camera-to-body extrinsic transform (4x4)
        self.T_cam_to_body = self._build_extrinsics(cam_height, cam_forward, tilt_deg)

        self._last_scan = np.zeros(GRID_NX * GRID_NY, dtype=np.float32)
        self._yaw_alpha = yaw_alpha
        self._yaw_rate_filtered = 0.0

    @staticmethod
    def _build_extrinsics(cam_height: float, cam_forward: float, tilt_deg: float) -> np.ndarray:
        """4x4 transform from camera frame to robot body frame."""
        tilt = np.radians(tilt_deg)
        R = np.array([
            [ np.cos(tilt), 0, np.sin(tilt)],
            [            0, 1,            0],
            [-np.sin(tilt), 0, np.cos(tilt)],
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = np.array([cam_forward, 0.0, cam_height])
        return T

    def start(self):
        err = self._camera.open(self._init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED Mini failed to open: {err}")
        print("[TerrainScanner] ZED Mini opened successfully")

    def stop(self):
        self._camera.close()
        print("[TerrainScanner] ZED Mini closed")

    def get_terrain_scan(self, robot_base_z: float = 0.0) -> np.ndarray:
        """
        Capture one frame and return terrain_scan[35].

        Args:
            robot_base_z: robot base height in world frame (metres).
                          Use 0.0 if unknown — heights will be relative
                          to the camera mounting height instead.

        Returns:
            np.ndarray shape (35,) — height of ground relative to robot base z.
            Positive = obstacle/step above base. Negative = drop below base.
            Zero = no valid depth at that grid cell.
        """
        if self._camera.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            return self._last_scan

        self._camera.retrieve_measure(self._point_cloud, sl.MEASURE.XYZRGBA)
        pts_raw = self._point_cloud.get_data()[:, :, :3]  # (H, W, 3)

        # Flatten and remove invalid (NaN/inf) points
        pts = pts_raw.reshape(-1, 3)
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]

        if len(pts) == 0:
            return self._last_scan

        # Transform from camera frame → robot body frame
        pts_body = (self.T_cam_to_body[:3, :3] @ pts.T).T + self.T_cam_to_body[:3, 3]

        # Sample minimum height at each of the 35 grid cells
        scan = np.zeros(GRID_NX * GRID_NY, dtype=np.float32)
        for i, gx in enumerate(GRID_X):
            for j, gy in enumerate(GRID_Y):
                mask = (
                    (np.abs(pts_body[:, 0] - gx) < SEARCH_RADIUS) &
                    (np.abs(pts_body[:, 1] - gy) < SEARCH_RADIUS)
                )
                if mask.any():
                    # Minimum z matches ray caster firing downward
                    scan[i * GRID_NY + j] = float(pts_body[mask, 2].min()) - robot_base_z

        self._last_scan = scan
        return scan

    def get_imu_data(self):
        """
        Returns (projected_gravity[3], yaw_rate) from ZED Mini IMU.

        projected_gravity: accelerometer reading in body frame (m/s²).
                           When upright and still: approx [0, 0, -9.81].
        yaw_rate:          rotation rate around vertical axis (rad/s),
                           low-pass filtered to reduce gyro flicker at rest.
        """
        self._camera.get_sensors_data(self._sensors_data, sl.TIME_REFERENCE.CURRENT)
        imu = self._sensors_data.get_imu_data()

        accel = imu.get_linear_acceleration()
        gyro  = imu.get_angular_velocity()

        projected_gravity = np.array([accel[0], accel[1], accel[2]], dtype=np.float32)

        # Low-pass filter: smooths gyro zero-rate flicker without adding much lag
        raw_yaw = float(gyro[2])
        self._yaw_rate_filtered = (
            self._yaw_alpha * raw_yaw + (1.0 - self._yaw_alpha) * self._yaw_rate_filtered
        )

        return projected_gravity, self._yaw_rate_filtered


# ---------------------------------------------------------------------------
# Standalone test — terminal + matplotlib heatmap visualisation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import argparse
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-height",   type=float, default=0.15,  help="Camera height above robot base (m)")
    parser.add_argument("--cam-forward",  type=float, default=0.05,  help="Camera forward offset (m)")
    parser.add_argument("--tilt-deg",     type=float, default=30.0,  help="Camera downward tilt (degrees)")
    parser.add_argument("--floor-offset", type=float, default=None,  help="Floor z offset for calibration (m). "
                                                                           "If not set, auto-calibrates from first 20 frames.")
    parser.add_argument("--no-plot",      action="store_true",        help="Terminal only, no matplotlib window")
    args = parser.parse_args()

    scanner = TerrainScanner(
        cam_height=args.cam_height,
        cam_forward=args.cam_forward,
        tilt_deg=args.tilt_deg,
    )
    scanner.start()

    print(f"Grid {GRID_NX}x{GRID_NY} | resolution={GRID_RES}m")
    print(f"Forward range : {GRID_X[0]:.2f} – {GRID_X[-1]:.2f} m")
    print(f"Lateral range : {GRID_Y[0]:.2f} – {GRID_Y[-1]:.2f} m")

    # Auto-calibrate floor offset if not provided
    floor_offset = args.floor_offset
    if floor_offset is None:
        print("Auto-calibrating floor offset from 20 frames (point at flat floor)...")
        samples = []
        for _ in range(20):
            s = scanner.get_terrain_scan(robot_base_z=0.0)
            valid = s[s != 0.0]
            if len(valid) > 0:
                samples.append(valid)
            time.sleep(0.05)
        if samples:
            floor_offset = float(np.median(np.concatenate(samples)))
            print(f"Floor offset: {floor_offset:.3f}m  (pass --floor-offset {floor_offset:.3f} to skip next time)")
        else:
            floor_offset = 0.0
            print("No valid points found, using floor_offset=0.0")

    print("Ctrl+C to stop\n")

    # Setup matplotlib heatmap
    if not args.no_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 8))
        dummy = np.zeros((GRID_NX, GRID_NY))
        im = ax.imshow(dummy, vmin=-0.3, vmax=0.3, cmap="RdYlGn_r",
                       aspect="auto", origin="lower")
        plt.colorbar(im, ax=ax, label="Height relative to floor (m)")
        ax.set_xlabel("Lateral (L ← → R)")
        ax.set_ylabel("Forward distance (m)")
        ax.set_xticks(range(GRID_NY))
        ax.set_xticklabels([f"{y:.2f}" for y in GRID_Y])
        ax.set_yticks(range(GRID_NX))
        ax.set_yticklabels([f"{x:.2f}" for x in GRID_X])
        ax.set_title("AWM terrain_scan — ZED Mini live")
        text_overlays = [[ax.text(j, i, "", ha="center", va="center",
                                  fontsize=8, color="black")
                          for j in range(GRID_NY)] for i in range(GRID_NX)]
        plt.tight_layout()

    try:
        while True:
            scan = scanner.get_terrain_scan(robot_base_z=floor_offset)
            gravity, yaw_rate = scanner.get_imu_data()
            grid = scan.reshape(GRID_NX, GRID_NY)

            # Terminal output
            print("\033[H\033[J", end="")
            print("terrain_scan — rows=forward distance, cols=left→right")
            print(f"floor_offset={floor_offset:.3f}m")
            print(f"{'':6s}  {'L':^7s}  {'':^7s}  {'C':^7s}  {'':^7s}  {'R':^7s}")
            for i in range(GRID_NX - 1, -1, -1):
                vals = "  ".join(f"{grid[i, j]:+.3f}" for j in range(GRID_NY))
                print(f"{GRID_X[i]:.2f}m  {vals}")
            print(f"\ngravity xyz : {gravity[0]:+.3f}  {gravity[1]:+.3f}  {gravity[2]:+.3f}")
            print(f"yaw_rate    : {yaw_rate:+.4f} rad/s")
            print(f"scan range  : {scan.min():+.3f} to {scan.max():+.3f} m")

            # Heatmap update
            if not args.no_plot:
                im.set_data(grid)
                for i in range(GRID_NX):
                    for j in range(GRID_NY):
                        text_overlays[i][j].set_text(f"{grid[i,j]:+.2f}")
                fig.canvas.draw()
                fig.canvas.flush_events()

            time.sleep(0.033)  # ~30 Hz

    except KeyboardInterrupt:
        print("\nStopped.")
        if not args.no_plot:
            plt.close()
        scanner.stop()
