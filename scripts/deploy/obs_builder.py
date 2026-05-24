"""
Assemble the 25-dim GRU+ProprioTorque observation vector.

Exact obs order matches AwmProprioTorqueCfg in awm_env_cfg.py:
  [0-1]   cmd_vel            [vx_cmd, ωz_cmd]
  [2]     base_lin_vel_x     wheel odometry
  [3]     base_ang_vel_z     ZED Mini gyro z
  [4-7]   wheel_velocities   [F_L, F_R, B_R, B_L] × sign_mask
  [8-11]  leg_positions      [F_L, F_R, B_L, B_R] rad (sim convention)
  [12-15] prev_leg_actions   last policy output indices [4:8]
  [16-18] projected_gravity  Madgwick output
  [19-20] progress_slip      EMA(step_progress), EMA(slip)  α=0.1
  [21-24] leg_torques        [F_L, F_R, B_L, B_R] / 2.70 Nm, clamped [-2, 2]

Obs normalizer (EmpiricalNormalization mean/std) is baked into the JIT policy —
no manual normalisation needed here.
"""

import numpy as np

WHEEL_RADIUS   = 0.0508        # m
WHEEL_SIGN     = np.array([1.0, -1.0, -1.0, 1.0])  # [F_L, F_R, B_R, B_L]
EMA_ALPHA      = 0.1
PROGRESS_MAX   = 0.5           # matches clamp in sim
OBS_DIM        = 25


class ObsBuilder:
    def __init__(self):
        self.prev_leg_actions = np.zeros(4, dtype=np.float32)
        self.prog_ema         = 0.0
        self.slip_ema         = 0.0
        self.prev_root_x      = None   # initialised on first call
        self._initialized     = False

    def reset(self) -> None:
        """Call when starting a new run (or after robot falls/resets)."""
        self.prev_leg_actions[:] = 0.0
        self.prog_ema   = 0.0
        self.slip_ema   = 0.0
        self.prev_root_x = None
        self._initialized = False

    def build(
        self,
        cmd_vel:          np.ndarray,   # (2,) [vx_cmd, ωz_cmd]
        wheel_vel_signed: np.ndarray,   # (4,) [F_L,F_R,B_R,B_L] rad/s, sign already applied
        leg_pos_rad:      np.ndarray,   # (4,) [F_L,F_R,B_L,B_R] sim-convention rad
        leg_torque_norm:  np.ndarray,   # (4,) [F_L,F_R,B_L,B_R] normalised, clamped
        projected_grav:   np.ndarray,   # (3,)
        ang_vel_z:        float,
        root_x:           float,        # integrated x position (from wheel odometry)
        dt:               float = 1/60,
    ) -> np.ndarray:
        """Returns float32 array of shape (25,)."""

        # ── base_lin_vel_x from wheel odometry ───────────────────────────────
        # Use raw wheel speeds (before sign mask) to get magnitude
        raw_wheel_speeds = wheel_vel_signed / WHEEL_SIGN   # undo sign mask
        base_vx = float(np.mean(np.abs(raw_wheel_speeds)) * WHEEL_RADIUS)

        # ── progress & slip EMA (matches sim exactly) ─────────────────────────
        if not self._initialized:
            self.prev_root_x = root_x
            self._initialized = True

        step_progress = float(np.clip(root_x - self.prev_root_x, 0.0, PROGRESS_MAX))
        mean_wheel_speed = float(np.mean(np.abs(raw_wheel_speeds)) * WHEEL_RADIUS)
        slip = float(max(mean_wheel_speed - abs(base_vx), 0.0))

        self.prog_ema = (1.0 - EMA_ALPHA) * self.prog_ema + EMA_ALPHA * step_progress
        self.slip_ema = (1.0 - EMA_ALPHA) * self.slip_ema + EMA_ALPHA * slip
        self.prev_root_x = root_x

        # ── assemble ──────────────────────────────────────────────────────────
        obs = np.empty(OBS_DIM, dtype=np.float32)
        obs[0:2]   = cmd_vel
        obs[2]     = base_vx
        obs[3]     = ang_vel_z
        obs[4:8]   = wheel_vel_signed
        obs[8:12]  = leg_pos_rad
        obs[12:16] = self.prev_leg_actions
        obs[16:19] = projected_grav
        obs[19]    = self.prog_ema
        obs[20]    = self.slip_ema
        obs[21:25] = leg_torque_norm

        return obs

    def update_prev_actions(self, leg_actions: np.ndarray) -> None:
        """Call with policy output indices [4:8] (leg commands) after each step."""
        self.prev_leg_actions[:] = leg_actions


# ── Unit test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    builder = ObsBuilder()

    # Dummy values — robot stationary, upright
    obs = builder.build(
        cmd_vel          = np.array([0.3, 0.0]),
        wheel_vel_signed = np.zeros(4),
        leg_pos_rad      = np.zeros(4),
        leg_torque_norm  = np.zeros(4),
        projected_grav   = np.array([0.0, 0.0, -1.0]),
        ang_vel_z        = 0.0,
        root_x           = 0.0,
    )

    assert obs.shape == (OBS_DIM,), f"Expected (25,), got {obs.shape}"
    assert obs.dtype == np.float32

    print(f"obs shape:  {obs.shape}  dtype: {obs.dtype}")
    labels = [
        "cmd_vx", "cmd_wz",
        "base_vx", "base_wz",
        "whl_FL", "whl_FR", "whl_BR", "whl_BL",
        "leg_FL", "leg_FR", "leg_BL", "leg_BR",
        "act_FL", "act_FR", "act_BL", "act_BR",
        "grav_x", "grav_y", "grav_z",
        "prog", "slip",
        "trq_FL", "trq_FR", "trq_BL", "trq_BR",
    ]
    for i, (label, val) in enumerate(zip(labels, obs)):
        print(f"  [{i:2d}] {label:<8} = {val:.4f}")
    print("ObsBuilder unit test passed.")
