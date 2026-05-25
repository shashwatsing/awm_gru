"""
AWM real-robot deployment — GRU+ProprioTorque policy at 60 Hz.

Usage (on Jetson or workstation with robot connected):
  python deploy.py --policy exported/policy.pt [--max_wheel_speed 2.0] [--log]

First-run checklist:
  1. Run dxl_interface.py standalone to verify motor IDs
  2. Run imu_reader.py standalone to verify projected_gravity ≈ [0,0,-1]
  3. Run obs_builder.py standalone to verify shape/dtype
  4. Run this script with --max_wheel_speed 2.0 (robot on table, wheels off ground)
  5. Verify obs values in log, check legs move to closed at cmd_vel=[0,0]
  6. Remove speed cap, place on flat ground, test with short forward command
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from dxl_interface import DxlInterface
from imu_reader import ImuReader
from obs_builder import ObsBuilder

HZ = 60
DT = 1.0 / HZ

# Sign mask for wheel velocity obs (undone inside DxlInterface — applied here for clarity)
WHEEL_SIGN = np.array([1.0, -1.0, -1.0, 1.0])


def decode_actions(
    actions: np.ndarray,
    max_wheel_speed: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    actions: (8,) float32 in [-1, 1] from policy.
    Returns:
        wheel_cmds_rad_s: (4,) [F_L, F_R, B_R, B_L]  rad/s in obs sign convention
        leg_extensions:   (4,) [F_L, F_R, B_L, B_R]  ∈ [0, 1]  (0=closed, 1=open)
    """
    wheel_cmds   = np.clip(actions[:4], -1.0, 1.0) * max_wheel_speed
    leg_ext      = np.clip(0.5 * actions[4:] + 0.0, 0.0, 1.0)
    return wheel_cmds, leg_ext


def integrate_odometry(x: float, wheel_vel_signed: np.ndarray, dt: float) -> float:
    """Integrate wheel odometry to get robot x position."""
    raw_speeds = wheel_vel_signed / WHEEL_SIGN
    vx = float(np.mean(np.abs(raw_speeds)) * 0.0508)
    return x + vx * dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="exported/policy.pt",
                        help="Path to JIT policy file")
    parser.add_argument("--max_wheel_speed", type=float, default=8.0,
                        help="Cap wheel speed in rad/s (use 2.0 for first run)")
    parser.add_argument("--cmd_vx", type=float, default=0.0,
                        help="Forward velocity command in m/s")
    parser.add_argument("--cmd_wz", type=float, default=0.0,
                        help="Yaw rate command in rad/s")
    parser.add_argument("--log", action="store_true",
                        help="Print obs/action values each step")
    args = parser.parse_args()

    # ── Load policy ───────────────────────────────────────────────────────────
    policy_path = Path(args.policy)
    if not policy_path.exists():
        raise FileNotFoundError(
            f"Policy not found: {policy_path}\n"
            "Run export_policy.py first, then copy exported/policy.pt here."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading policy from {policy_path} on {device}...")
    policy = torch.jit.load(str(policy_path), map_location=device)
    policy.eval()
    print("Policy loaded.")

    # ── Init hardware ─────────────────────────────────────────────────────────
    dxl = DxlInterface()
    imu = ImuReader()
    obs_builder = ObsBuilder()

    cmd_vel = np.array([args.cmd_vx, args.cmd_wz], dtype=np.float32)
    root_x  = 0.0

    if args.max_wheel_speed < 8.0:
        print(f"[SAFETY] Wheel speed capped at {args.max_wheel_speed:.1f} rad/s")

    print(f"\nStarting 60 Hz loop — Ctrl-C to stop")
    print(f"cmd_vel = [{args.cmd_vx:.2f} m/s, {args.cmd_wz:.2f} rad/s]")

    loop_count = 0
    overrun_count = 0

    try:
        while True:
            t0 = time.monotonic()

            # 1. Read sensors (wheel + leg in parallel, then IMU)
            wheel_vel, leg_pos, leg_torque = dxl.read_all()
            grav, ang_vel_z, acc_world_x = imu.read()

            # 2. Integrate odometry
            root_x = integrate_odometry(root_x, wheel_vel, DT)

            # 3. Build 25-dim obs
            obs = obs_builder.build(
                cmd_vel          = cmd_vel,
                wheel_vel_signed = wheel_vel,
                leg_pos_rad      = leg_pos,
                leg_torque_norm  = leg_torque,
                projected_grav   = grav,
                ang_vel_z        = ang_vel_z,
                root_x           = root_x,
                acc_world_x      = acc_world_x,
                dt               = DT,
            )

            # 4. Policy inference — GRU hidden state managed internally by JIT model
            with torch.no_grad():
                obs_t   = torch.from_numpy(obs).unsqueeze(0).to(device)
                act_t   = policy(obs_t)
            actions = act_t.squeeze(0).cpu().numpy()

            # 5. Decode actions
            wheel_cmds, leg_ext = decode_actions(actions, args.max_wheel_speed)

            # 6. Write commands
            dxl.write_wheel_velocities(wheel_cmds)
            dxl.write_leg_positions(leg_ext)

            # 7. Store leg actions for next obs
            obs_builder.update_prev_actions(actions[4:])

            # 8. Logging
            if args.log and loop_count % 12 == 0:  # ~5 Hz print rate
                vx_est = float(np.mean(np.abs(wheel_vel / WHEEL_SIGN)) * 0.0508)
                print(
                    f"[{loop_count:6d}] "
                    f"vx={vx_est:.3f} "
                    f"whl_cmd=[{wheel_cmds[0]:+.2f},{wheel_cmds[1]:+.2f},"
                    f"{wheel_cmds[2]:+.2f},{wheel_cmds[3]:+.2f}] "
                    f"grav=[{grav[0]:+.2f},{grav[1]:+.2f},{grav[2]:+.2f}] "
                    f"leg_ext=[{leg_ext[0]:.2f},{leg_ext[1]:.2f},"
                    f"{leg_ext[2]:.2f},{leg_ext[3]:.2f}] "
                    f"trq=[{leg_torque[0]:+.2f},{leg_torque[1]:+.2f},"
                    f"{leg_torque[2]:+.2f},{leg_torque[3]:+.2f}]"
                )

            loop_count += 1

            # 9. Rate control
            elapsed = time.monotonic() - t0
            remaining = DT - elapsed
            if remaining > 0:
                time.sleep(remaining)
            else:
                overrun_count += 1
                if overrun_count % 60 == 1:
                    print(f"[WARN] Loop overrun: {elapsed*1000:.1f} ms (#{overrun_count})")

    except KeyboardInterrupt:
        print(f"\nStopping after {loop_count} steps ({overrun_count} overruns).")

    finally:
        print("Zeroing wheel commands and closing...")
        dxl.write_wheel_velocities(np.zeros(4))
        dxl.close()
        imu.close()


if __name__ == "__main__":
    main()
