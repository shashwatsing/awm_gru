"""Synthetic torque sweep — tests whether the GRU learned the torque→extend/retract mapping.

No simulator needed. Loads the actor GRU + MLP from a checkpoint and runs a 5-second
open-loop inference with a fabricated 25-dim observation where leg torques are swept
from 0 (flat terrain) → 0.7 (rough terrain) → 0 (flat again).

Observation layout (AwmProprioTorqueCfg, 25 dims):
  [0:2]   commanded_velocity  [vx=0.3, yaw=0.0]
  [2]     base_lin_vel_x      0.3 m/s
  [3]     base_ang_vel_z      0.0 rad/s
  [4:8]   wheel_velocities    ~5.9 rad/s (0.3 m/s / 0.0508 m radius)
  [8:12]  leg_positions       -2.53 rad (closed, fixed)
  [12:16] leg_actions         previous policy leg output (tracked)
  [16:19] projected_gravity   [0, 0, -1]
  [19:21] progress_slip_hist  [0.01, 0.0]
  [21:25] leg_torques         SWEPT (normalized by 2.70 Nm)

Action layout (8 dims, from AwmDriveAction):
  [0:4]  wheel velocity commands  (not plotted)
  [4:8]  leg position commands    leg_cmd in [-1, 1]
         extension = clamp(0.5 * leg_cmd + 0.0, 0, 1)
         -1 → fully closed, +1 → fully open
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    default="logs/rsl_rl/awm_gru_proprio_torque/2026-05-06_16-43-59/model_best.pt",
)
parser.add_argument("--hz", type=float, default=60.0)
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--peak-torque", type=float, default=1.2)
args = parser.parse_args()

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt = torch.load(args.checkpoint, map_location="cpu")
sd = ckpt["model_state_dict"]
print(f"Loaded checkpoint iter={ckpt['iter']}  path={args.checkpoint}")

# ── Build actor GRU ───────────────────────────────────────────────────────────
gru = torch.nn.GRU(input_size=25, hidden_size=256, num_layers=1, batch_first=True)
gru.weight_ih_l0.data = sd["memory_a.rnn.weight_ih_l0"]
gru.weight_hh_l0.data = sd["memory_a.rnn.weight_hh_l0"]
gru.bias_ih_l0.data   = sd["memory_a.rnn.bias_ih_l0"]
gru.bias_hh_l0.data   = sd["memory_a.rnn.bias_hh_l0"]

# ── Build actor MLP: Linear(256,256)-ELU-Linear(256,128)-ELU-Linear(128,8) ──
actor = torch.nn.Sequential(
    torch.nn.Linear(256, 256), torch.nn.ELU(),
    torch.nn.Linear(256, 128), torch.nn.ELU(),
    torch.nn.Linear(128, 8),
)
actor[0].weight.data = sd["actor.0.weight"]
actor[0].bias.data   = sd["actor.0.bias"]
actor[2].weight.data = sd["actor.2.weight"]
actor[2].bias.data   = sd["actor.2.bias"]
actor[4].weight.data = sd["actor.4.weight"]
actor[4].bias.data   = sd["actor.4.bias"]

gru.eval()
actor.eval()

# ── Torque sweep schedule — step function with 3 clear phases ─────────────────
# flat(3s) → rough(4s) → flat(3s)  — each phase long enough for GRU to settle
T = int(args.duration * args.hz)
t_flat1 = int(3.0 * args.hz)
t_rough  = int(4.0 * args.hz)

torque_profile = np.zeros(T)
for i in range(T):
    if t_flat1 <= i < t_flat1 + t_rough:
        torque_profile[i] = args.peak_torque

# ── Inference loop ────────────────────────────────────────────────────────────
hidden = torch.zeros(1, 1, 256)  # (num_layers, batch, hidden)
leg_actions_prev = np.zeros(4)

times       = np.arange(T) / args.hz
leg_cmds    = np.zeros((T, 4))   # raw policy output for legs [-1, 1]
extensions  = np.zeros((T, 4))   # derived extension [0, 1]

LEG_CLOSED     = -2.53   # lower joint limit (closed)
LEG_OPEN       =  0.0    # upper joint limit (fully extended) — approx from USD
WHEEL_VEL_FWD  = 0.3 / 0.0508   # 0.3 m/s / wheel radius

leg_pos = np.array([LEG_CLOSED] * 4, dtype=np.float32)  # start closed
dt = 1.0 / args.hz
# Leg position tracks command with ~0.1s time constant (high-gain sim actuator)
leg_tau = 0.05  # seconds
leg_alpha = dt / (leg_tau + dt)

# ── Warmup: run GRU for 3s on stable flat terrain so hidden state settles ────
WARMUP = int(3.0 * args.hz)
with torch.no_grad():
    for _ in range(WARMUP):
        obs_w = np.array([
            0.3, 0.0, 0.3, 0.0,
            WHEEL_VEL_FWD, WHEEL_VEL_FWD, WHEEL_VEL_FWD, WHEEL_VEL_FWD,
            *leg_pos, *leg_actions_prev,
            0.0, 0.0, -1.0, 0.01, 0.0,
            0.0, 0.0, 0.0, 0.0,  # flat — torque = 0
        ], dtype=np.float32)
        obs_t = torch.tensor(obs_w).unsqueeze(0).unsqueeze(0)
        gru_out, hidden = gru(obs_t, hidden)
        action = torch.clamp(actor(gru_out.squeeze(0)).squeeze(0), -1.0, 1.0).numpy()
        leg_cmd = action[4:]
        leg_actions_prev = leg_cmd
        ext = np.clip(0.5 * leg_cmd, 0.0, 1.0)
        leg_target = LEG_CLOSED + ext * (LEG_OPEN - LEG_CLOSED)
        leg_pos = leg_pos + leg_alpha * (leg_target - leg_pos)

with torch.no_grad():
    for i in range(T):
        torq = torque_profile[i]

        obs = np.array([
            0.3,  0.0,                              # commanded_velocity
            0.3,                                    # base_lin_vel_x
            0.0,                                    # base_ang_vel_z
            WHEEL_VEL_FWD, WHEEL_VEL_FWD,
            WHEEL_VEL_FWD, WHEEL_VEL_FWD,          # wheel_velocities
            *leg_pos,                               # leg_positions (tracks commands)
            *leg_actions_prev,                       # leg_actions (previous)
            0.0,  0.0, -1.0,                        # projected_gravity
            0.01, 0.0,                              # progress_slip_hist
            torq, torq, torq, torq,                 # leg_torques (all 4 legs same)
        ], dtype=np.float32)

        obs_t = torch.tensor(obs).unsqueeze(0).unsqueeze(0)  # (1, 1, 25)
        gru_out, hidden = gru(obs_t, hidden)                  # (1, 1, 256)
        action = actor(gru_out.squeeze(0)).squeeze(0)         # (8,)
        action_clamped = torch.clamp(action, -1.0, 1.0).numpy()

        leg_cmd = action_clamped[4:]   # last 4 = legs
        leg_actions_prev = leg_cmd

        # First-order lag on leg position (matches high-gain actuator in sim)
        ext = np.clip(0.5 * leg_cmd + 0.0, 0.0, 1.0)
        leg_target = LEG_CLOSED + ext * (LEG_OPEN - LEG_CLOSED)
        leg_pos = leg_pos + leg_alpha * (leg_target - leg_pos)

        leg_cmds[i]   = leg_cmd
        extensions[i] = ext

# ── Plot ──────────────────────────────────────────────────────────────────────
leg_labels = ["FL", "FR", "BL", "BR"]
colors     = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"]

fig = plt.figure(figsize=(12, 8))
gs  = gridspec.GridSpec(3, 1, hspace=0.45)

# Input torques
ax0 = fig.add_subplot(gs[0])
ax0.plot(times, torque_profile, color="gray", linewidth=2)
ax0.set_ylabel("Torque input\n(normalised)", fontsize=11)
ax0.set_title("Synthetic Torque Sweep — GRU Policy Response", fontsize=13, fontweight="bold")
ax0.set_ylim(-0.05, 1.05)
ax0.axhline(0, color="k", linewidth=0.5, linestyle="--")
ax0.fill_between(times, 0, torque_profile, alpha=0.15, color="gray")
ax0.set_xticklabels([])

def smooth(x, w=30):
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")

# Leg command output
ax1 = fig.add_subplot(gs[1])
for j, (lbl, col) in enumerate(zip(leg_labels, colors)):
    ax1.plot(times, leg_cmds[:, j], color=col, linewidth=0.6, alpha=0.3)
    ax1.plot(times, smooth(leg_cmds[:, j]), label=lbl, color=col, linewidth=2.0)
ax1.axhline(0, color="k", linewidth=0.5, linestyle="--")
ax1.set_ylabel("Leg cmd (raw)\n[−1=closed, +1=open]", fontsize=11)
ax1.legend(loc="upper right", ncol=4, fontsize=9)
ax1.set_ylim(-1.1, 1.1)
ax1.set_xticklabels([])

# Extension [0,1]
ax2 = fig.add_subplot(gs[2])
for j, (lbl, col) in enumerate(zip(leg_labels, colors)):
    ax2.plot(times, extensions[:, j], color=col, linewidth=0.6, alpha=0.3)
    ax2.plot(times, smooth(extensions[:, j]), label=lbl, color=col, linewidth=2.0)
ax2.axhline(0.5, color="k", linewidth=0.5, linestyle="--", label="half-open")
ax2.set_ylabel("Extension\n[0=closed, 1=open]", fontsize=11)
ax2.set_xlabel("Time (s)", fontsize=11)
ax2.legend(loc="upper right", ncol=4, fontsize=9)
ax2.set_ylim(-0.05, 1.05)

# Shade phases
for ax in [ax0, ax1, ax2]:
    t_r0 = t_flat1 / args.hz
    t_r1 = (t_flat1 + t_rough) / args.hz
    ax.axvspan(t_r0, t_r1, alpha=0.07, color="red", label="_rough phase")
    ax.axvline(t_r0, color="red", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.axvline(t_r1, color="blue", linewidth=1.0, linestyle="--", alpha=0.6)

plt.savefig("torque_sweep.png", dpi=150, bbox_inches="tight")
print("Saved: torque_sweep.png")
plt.show()
