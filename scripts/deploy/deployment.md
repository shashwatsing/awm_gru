# Real Robot Deployment Plan — AWM GRU Policy

## Context

The GRU+ProprioTorque policy (25-dim obs, no camera) is trained in Isaac Lab (awm_gru repo) and needs to run on the physical AWM robot at 60 Hz. The robot has 4 XM430-W210-R wheel motors (RS-485) and 4 XM430-W350-T leg motors (TTL, mode 5 current-based position control), a ZED Mini for IMU, and a Jetson Orin Nano Super for inference. Nothing is written yet — this plan builds the entire pipeline from scratch.

---

## File Structure

All files go in `/home/shashwat/awm_gru/scripts/deploy/`:

```
scripts/deploy/
├── export_policy.py      # Step 1: export JIT model from checkpoint (run on workstation)
├── dxl_interface.py      # Step 2: Dynamixel sync reads + sync writes
├── imu_reader.py         # Step 3: ZED Mini IMU → projected_gravity + ang_vel_z
├── obs_builder.py        # Step 4: assemble 25-dim obs tensor
└── deploy.py             # Step 5: main 60 Hz inference loop
```

After dev on workstation, copy `scripts/deploy/` + `exported/policy.pt` to Jetson.

---

## Critical Reference Values

### Observation Space (25 dims, EXACT ORDER)

| Idx   | Term               | Dims | Notes |
|-------|--------------------|------|-------|
| 0–1   | cmd_vel            | 2    | [vx_cmd, ωz_cmd] from joystick |
| 2     | base_lin_vel_x     | 1    | wheel odometry |
| 3     | base_ang_vel_z     | 1    | ZED Mini gyro z (raw) |
| 4–7   | wheel_velocities   | 4    | [F_L, F_R, B_R, B_L] × signs [+1,−1,−1,+1] |
| 8–11  | leg_positions      | 4    | [F_L, F_R, B_L, B_R] in rad (sim convention) |
| 12–15 | prev_leg_actions   | 4    | last policy output [4:8] (leg only) |
| 16–18 | projected_gravity  | 3    | Madgwick → rotate [0,0,−1] into body frame |
| 19–20 | progress_slip      | 2    | EMA α=0.1 |
| 21–24 | leg_torques        | 4    | [F_L, F_R, B_L, B_R] / 2.70 Nm, clamp [−2,2] |

**Obs normalizer**: EmpiricalNormalization (mean/std) baked into JIT export — no manual normalization needed in deploy code.

### Action Space (8 dims)

| Idx  | Joint             | Scale |
|------|-------------------|-------|
| 0–3  | wheel_F_L/F_R/B_R/B_L | × 8.0 rad/s |
| 4–7  | leg_F_L/F_R/B_L/B_R   | clamp(0.5×a, 0, 1) → extension [0,1] |

### Conversion Formulas

```python
# Wheel DXL velocity → rad/s
wheel_rad_s = signed_value * 0.229 * 2 * pi / 60  # XM430-W210-R unit = 0.229 RPM/LSB

# Wheel odometry → base vx
vx = mean(abs(wheel_rad_s)) * 0.0508  # wheel_radius = 0.0508 m

# Leg DXL ticks → sim joint angle (rad)
# Left legs (F_L, B_L): closed=3641 ticks, open=1631 ticks
sim_rad_left  = -(3641 - ticks) * pi / 2010   # 0 at closed, -π at open
# Right legs (F_R, B_R): closed=1631 ticks, open=3641 ticks
sim_rad_right = (ticks - 1631) * pi / 2010    # 0 at closed, +π at open

# Leg extension → DXL goal ticks
# Left legs:  ticks = round(3641 - extension * 2010)
# Right legs: ticks = round(1631 + extension * 2010)
# extension = clamp(0.5 * action + 0.0, 0.0, 1.0)

# Leg torque from current
torque_Nm   = current_LSB * 2.69e-3 * 3.83     # calibrated 2026-04-19
torque_norm = clamp(torque_Nm / 2.70, -2.0, 2.0)

# Wheel cmd → DXL goal velocity
goal_vel_LSB = round(wheel_rad_s * 60 / (0.229 * 2 * pi))
```

### Dynamixel Register Addresses (XM430, Protocol 2.0)

| Register            | Addr | Size | Bus  |
|---------------------|------|------|------|
| Present Current     | 126  | 2B   | TTL  |
| Present Velocity    | 128  | 4B   | Both |
| Present Position    | 132  | 4B   | TTL  |
| Goal Velocity       | 104  | 4B   | RS-485 |
| Goal Position       | 116  | 4B   | TTL  |
| Torque Enable       | 64   | 1B   | Both |

Baud: 1,000,000. FTDI latency timer: 1ms (udev rule already installed).

---

## Step-by-Step Implementation

### Step 1 — Export Policy (`export_policy.py`)

Run on workstation. `play.py` already exports on startup, so this is a thin wrapper:

```python
# export_policy.py — run once on workstation
import subprocess, sys

checkpoint = sys.argv[1]  # path to model_best.pt
subprocess.run([
    "conda", "run", "-n", "env_isaaclab",
    "python", "scripts/rsl_rl/play.py",
    "--task", "Template-Awm_GRU_ProprioTorque-v0",
    "--num_envs", "1",
    "--checkpoint", checkpoint,
    "--headless",
], check=True)
# Output: <checkpoint_dir>/exported/policy.pt and policy.onnx
```

Copy `exported/policy.pt` to Jetson. The JIT model includes obs normalizer and GRU hidden state — no extra files needed.

---

### Step 2 — Dynamixel Interface (`dxl_interface.py`)

Uses `GroupSyncRead` and `GroupSyncWrite` from `dynamixel_sdk` for 60 Hz reads.

```python
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncRead, GroupSyncWrite, COMM_SUCCESS
import numpy as np

WHEEL_PORT = "/dev/ttyUSB0"   # RS-485, XM430-W210-R ×4
LEG_PORT   = "/dev/ttyUSB1"   # TTL,    XM430-W350-T ×4
BAUDRATE   = 1_000_000
PROTOCOL   = 2.0

WHEEL_IDS  = [1, 2, 3, 4]    # F_L, F_R, B_R, B_L — verify with robot
LEG_IDS    = [5, 6, 7, 8]    # F_L, F_R, B_L, B_R — verify with robot

ADDR_PRESENT_VELOCITY = 128   # 4B signed
ADDR_PRESENT_CURRENT  = 126   # 2B signed
ADDR_PRESENT_POSITION = 132   # 4B signed
ADDR_GOAL_VELOCITY    = 104   # 4B signed
ADDR_GOAL_POSITION    = 116   # 4B signed

class DxlInterface:
    def __init__(self):
        # Open both ports, set baud, enable torque
        # Init GroupSyncRead for wheels (velocity) and legs (current + position)
        # Init GroupSyncWrite for wheel velocity + leg position

    def read_wheel_velocities(self) -> np.ndarray:
        # GroupSyncRead from WHEEL_IDS, addr=128, len=4
        # Returns [F_L, F_R, B_R, B_L] in rad/s (signed, with sign mask applied)

    def read_leg_state(self) -> tuple[np.ndarray, np.ndarray]:
        # GroupSyncRead from LEG_IDS, addr=126, len=10  (covers current+position contiguously)
        # Returns (leg_positions_rad [F_L,F_R,B_L,B_R], leg_torques_norm [F_L,F_R,B_L,B_R])

    def write_wheel_velocities(self, vel_rad_s: np.ndarray):
        # Convert rad/s → LSB, GroupSyncWrite

    def write_leg_positions(self, extensions: np.ndarray):
        # extensions: [F_L, F_R, B_L, B_R] ∈ [0,1]
        # Left legs  (idx 0,2): ticks = round(3641 - ext * 2010), clamp [1631, 3641]
        # Right legs (idx 1,3): ticks = round(1631 + ext * 2010), clamp [1631, 3641]
        # GroupSyncWrite

    def close(self): ...
```

**NOTE**: `WHEEL_IDS` and `LEG_IDS` must be confirmed by scanning the bus before first use (`python -c "from dxl_interface import DxlInterface; DxlInterface().scan()"`).

---

### Step 3 — IMU Reader (`imu_reader.py`)

```python
import pyzed.sl as sl
import imufusion       # pip install imufusion
import numpy as np

class ImuReader:
    def __init__(self, sample_rate=200):
        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.sdk_verbose = 0
        self.zed.open(init)
        # Do NOT call enable_positional_tracking — raw IMU only
        self.ahrs = imufusion.Ahrs()
        self.ahrs.settings = imufusion.Settings(
            imufusion.Convention.NWU, 0.5, 2000, 10, 10, 5 * sample_rate
        )
        self.dt = 1.0 / sample_rate
        self.sensors_data = sl.SensorsData()

    def read(self) -> tuple[np.ndarray, float]:
        """Returns (projected_gravity [3], ang_vel_z float)."""
        self.zed.get_sensors_data(self.sensors_data, sl.TIME_REFERENCE.CURRENT)
        imu = self.sensors_data.get_imu_data()
        acc = imu.get_linear_acceleration()      # [ax, ay, az] m/s²
        gyr = imu.get_angular_velocity()         # [gx, gy, gz] deg/s

        # Madgwick update
        self.ahrs.update_no_magnetometer(
            np.array([gyr.x, gyr.y, gyr.z]),
            np.array([acc.x, acc.y, acc.z]),
            self.dt
        )
        q = self.ahrs.quaternion.wxyz            # [w, x, y, z]

        # Rotate [0,0,-1] (gravity world) into body frame
        # projected_gravity = R^T @ [0,0,-1]  where R is body→world rotation from q
        w, x, y, z = q
        gx = 2*(x*z - w*y)           # x component of rotated gravity
        gy = 2*(y*z + w*x)           # y component
        gz = w*w - x*x - y*y + z*z   # z component (negative for upright)
        projected_gravity = np.array([-gx, -gy, -gz])  # negate: gravity points down

        ang_vel_z = np.radians(gyr.z)            # gyro z in rad/s
        return projected_gravity, ang_vel_z

    def close(self):
        self.zed.close()
```

---

### Step 4 — Obs Builder (`obs_builder.py`)

```python
import numpy as np

WHEEL_SIGN_MASK = np.array([1.0, -1.0, -1.0, 1.0])  # [F_L, F_R, B_R, B_L]
PROGRESS_ALPHA  = 0.1
SLIP_ALPHA      = 0.1
WHEEL_RADIUS    = 0.0508  # m

class ObsBuilder:
    def __init__(self):
        self.prev_leg_actions = np.zeros(4)
        self.progress_ema     = 0.0
        self.slip_ema         = 0.0
        self.prev_x           = 0.0   # integrated x from odometry

    def build(self,
              cmd_vel:        np.ndarray,   # [vx_cmd, ωz_cmd]
              wheel_vel_rad:  np.ndarray,   # [F_L, F_R, B_R, B_L] rad/s raw
              leg_pos_rad:    np.ndarray,   # [F_L, F_R, B_L, B_R] in sim convention
              leg_torque_norm: np.ndarray,  # [F_L, F_R, B_L, B_R] normalized
              projected_grav: np.ndarray,   # [3]
              ang_vel_z:      float,
              dt:             float = 1/60,
              ) -> np.ndarray:

        # base_lin_vel_x from odometry
        vx = np.mean(np.abs(wheel_vel_rad)) * WHEEL_RADIUS

        # wheel obs: apply sign mask
        wheel_vel_obs = wheel_vel_rad * WHEEL_SIGN_MASK

        # progress & slip EMA
        dx = vx * dt
        self.prev_x += dx
        slip = np.mean(np.abs(wheel_vel_rad)) * WHEEL_RADIUS - vx  # 0 when no slip
        self.progress_ema = (1 - PROGRESS_ALPHA) * self.progress_ema + PROGRESS_ALPHA * vx
        self.slip_ema     = (1 - SLIP_ALPHA)     * self.slip_ema     + SLIP_ALPHA     * slip

        obs = np.concatenate([
            cmd_vel,                          # [0-1]
            [vx],                             # [2]
            [ang_vel_z],                      # [3]
            wheel_vel_obs,                    # [4-7]
            leg_pos_rad,                      # [8-11]
            self.prev_leg_actions,            # [12-15]
            projected_grav,                   # [16-18]
            [self.progress_ema, self.slip_ema], # [19-20]
            leg_torque_norm,                  # [21-24]
        ]).astype(np.float32)
        assert obs.shape == (25,)
        return obs

    def update_prev_actions(self, leg_actions: np.ndarray):
        """Call after each policy step with the leg portion of actions (indices 4:8)."""
        self.prev_leg_actions = leg_actions.copy()
```

---

### Step 5 — Main 60 Hz Loop (`deploy.py`)

```python
import torch, time, numpy as np
from dxl_interface import DxlInterface
from imu_reader import ImuReader
from obs_builder import ObsBuilder

POLICY_PATH = "exported/policy.pt"
HZ          = 60
DT          = 1.0 / HZ

def decode_actions(actions: np.ndarray):
    """actions: (8,) float32 in [-1, 1]"""
    wheel_cmds = actions[:4] * 8.0                            # rad/s
    leg_ext    = np.clip(0.5 * actions[4:] + 0.0, 0.0, 1.0) # [0,1]
    return wheel_cmds, leg_ext

def main():
    policy = torch.jit.load(POLICY_PATH, map_location="cuda")
    policy.eval()

    dxl = DxlInterface()
    imu = ImuReader()
    obs_builder = ObsBuilder()

    cmd_vel = np.array([0.0, 0.0], dtype=np.float32)  # set from joystick

    print("Starting 60 Hz loop — Ctrl-C to stop")
    try:
        while True:
            t0 = time.monotonic()

            # 1. Read sensors
            wheel_vel_raw = dxl.read_wheel_velocities()        # [F_L,F_R,B_R,B_L] rad/s
            leg_pos, leg_torque = dxl.read_leg_state()        # rad, normalized
            grav, ang_vel_z = imu.read()

            # 2. Build obs (25-dim)
            obs = obs_builder.build(cmd_vel, wheel_vel_raw, leg_pos,
                                     leg_torque, grav, ang_vel_z, dt=DT)

            # 3. Inference (GRU hidden state managed internally by JIT model)
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0).to("cuda")
                actions_t = policy(obs_t)
            actions = actions_t.squeeze(0).cpu().numpy()

            # 4. Decode + write
            wheel_cmds, leg_ext = decode_actions(actions)
            dxl.write_wheel_velocities(wheel_cmds)
            dxl.write_leg_positions(leg_ext)

            # 5. Update prev actions buffer (leg part only)
            obs_builder.update_prev_actions(actions[4:])

            # 6. Rate control
            elapsed = time.monotonic() - t0
            if elapsed < DT:
                time.sleep(DT - elapsed)
            else:
                print(f"[WARN] Loop overrun: {elapsed*1000:.1f} ms")

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        dxl.write_wheel_velocities(np.zeros(4))
        dxl.close()
        imu.close()

if __name__ == "__main__":
    main()
```

---

## Calibration Required Before First Run

The tick↔rad mapping for legs assumes:
- Left legs (F_L, B_L): closed = 3641 ticks, open = 1631 ticks
- Right legs (F_R, B_R): closed = 1631 ticks, open = 3641 ticks

**Must verify on real robot before running policy:**
1. Command each leg to 0 extension (closed) → read ticks → confirm ~3641 (left) / ~1631 (right)
2. Command each leg to 1.0 extension (open) → read ticks → confirm ~1631 (left) / ~3641 (right)
3. If off by more than ±50 ticks, update the constants in `dxl_interface.py`

Also verify DXL IDs match joint labels (`WHEEL_IDS`, `LEG_IDS`) by commanding one motor at a time.

---

## Jetson Setup

```bash
# On Jetson Orin Nano Super (JetPack 6.1)
pip install imufusion dynamixel_sdk

# Copy from workstation
scp -r /home/shashwat/awm_gru/scripts/deploy/ jetson:~/awm_deploy/
scp <checkpoint_dir>/exported/policy.pt jetson:~/awm_deploy/exported/

# Set FTDI latency (already has udev rule — verify it's installed)
cat /etc/udev/rules.d/99-dynamixel.rules
```

---

## Build Order

1. `export_policy.py` — run on workstation, verify `exported/policy.pt` exists
2. `dxl_interface.py` — test on workstation with USB connected: scan IDs, read state
3. `imu_reader.py` — test on workstation or Jetson: verify projected_gravity ≈ [0,0,-1] when flat
4. `obs_builder.py` — unit test: feed dummy values, assert shape (25,) and index values
5. `deploy.py` — first run on workstation with motors enabled but `max_wheel_speed` capped to 2.0 rad/s

---

## First Run Protocol

1. Robot on table, wheels not touching ground
2. Cap wheel speed in `decode_actions`: `wheel_cmds = np.clip(wheel_cmds, -2.0, 2.0)`
3. Run for 10 s, log obs and actions — verify obs values are in reasonable ranges
4. Check legs move slowly and return to closed when `cmd_vel = [0, 0]`
5. Remove speed cap, place on flat ground, short forward command

---

## Files to Create

| File | New | Dependencies |
|------|-----|--------------|
| `scripts/deploy/export_policy.py` | New | `play.py` (existing) |
| `scripts/deploy/dxl_interface.py` | New | `dynamixel_sdk` |
| `scripts/deploy/imu_reader.py` | New | `pyzed`, `imufusion` |
| `scripts/deploy/obs_builder.py` | New | `numpy` |
| `scripts/deploy/deploy.py` | New | All above + `torch` |

Existing reuse:
- `awm_transformer/scripts/dynamixel/dxl_utils.py` — latency timer function, register addresses
- `awm_gru/scripts/rsl_rl/play.py` — policy export (already handles JIT + obs normalizer)
