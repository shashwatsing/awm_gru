"""Plot torque calibration results from characterize_torque.py JSON output."""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CURRENT_UNIT = 2.69   # mA per LSB
TORQUE_CONSTANT_DATASHEET = 1.78  # Nm/A

def load(path: str):
    return json.loads(Path(path).read_text())

def main():
    if len(sys.argv) < 2:
        # auto-find latest
        files = sorted(Path("/home/shashwat/awm_transformer").glob("torque_cal_*.json"))
        if not files:
            raise FileNotFoundError("No torque_cal_*.json found")
        path = str(files[-1])
    else:
        path = sys.argv[1]

    print(f"Loading: {path}")
    data = load(path)
    trials = data["trials"]

    arms      = np.array([t["arm_m"]          for t in trials])
    weights   = np.array([t["weight_kg"]       for t in trials])
    theory_nm = np.array([t["theoretical_nm"]  for t in trials])
    curr_lsb  = np.array([t["mean_current_mA"] for t in trials])  # raw LSB
    curr_std  = np.array([t["std_current_mA"]  for t in trials])
    curr_ma   = np.abs(curr_lsb) * CURRENT_UNIT   # actual mA
    curr_a    = curr_ma / 1000.0                   # actual A

    # Best-fit torque constant
    # torque = K * current_A  →  K = mean(torque / current_A)
    K_fit = np.mean(theory_nm / curr_a)
    K_std = np.std(theory_nm / curr_a)

    # Group by (arm, weight) and compute mean/std across 3 trials
    groups = {}
    for t in trials:
        key = (t["arm_m"], t["weight_kg"])
        if key not in groups:
            groups[key] = []
        groups[key].append(np.abs(t["mean_current_mA"]) * CURRENT_UNIT)  # actual mA

    group_keys    = sorted(groups.keys())
    group_theory  = np.array([k[0] * k[1] * 9.81 for k in group_keys])
    group_mean_ma = np.array([np.mean(groups[k]) for k in group_keys])
    group_std_ma  = np.array([np.std(groups[k])  for k in group_keys])
    group_arms    = np.array([k[0] for k in group_keys])
    group_weights = np.array([k[1] for k in group_keys])

    colors  = {0.04: '#1f77b4', 0.08: '#ff7f0e', 0.12: '#2ca02c'}
    markers = {0.5: 'o', 1.0: 's'}
    arm_labels = {0.04: '4 cm', 0.08: '8 cm', 0.12: '12 cm'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"XM430-W350-T Torque Calibration  |  Motor ID {data['motor_id']}  |  {data['timestamp'][:10]}\n"
                 f"Each point = mean ± std of 3 trials",
                 fontsize=12, fontweight='bold')

    # ── Plot 1: Current (mA) vs Theoretical Torque ──────────────────────────
    ax1 = axes[0]
    for arm in sorted(colors):
        for wt in sorted(markers):
            mask = (group_arms == arm) & (group_weights == wt)
            if not mask.any():
                continue
            ax1.errorbar(group_theory[mask], group_mean_ma[mask],
                         yerr=group_std_ma[mask],
                         fmt=markers[wt], color=colors[arm],
                         capsize=5, markersize=8, linewidth=1.5,
                         label=f'{arm_labels[arm]}, {wt}kg')

    x_fit = np.linspace(0, group_theory.max() * 1.1, 100)
    ax1.plot(x_fit, x_fit / K_fit * 1000,
             'k--', linewidth=2, label=f'fit K={K_fit:.2f} Nm/A')
    ax1.plot(x_fit, x_fit / TORQUE_CONSTANT_DATASHEET * 1000,
             'r:', linewidth=2, label=f'datasheet K={TORQUE_CONSTANT_DATASHEET} Nm/A')

    ax1.set_xlabel('Theoretical torque (Nm)', fontsize=11)
    ax1.set_ylabel('Measured current (mA)', fontsize=11)
    ax1.set_title('Current vs Torque')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Torque constant per group ────────────────────────────────────
    ax2 = axes[1]
    group_K_mean = group_theory / (group_mean_ma / 1000)
    # Propagate std: K = T / I, δK/K = δI/I → δK = K * δI/I
    group_K_std  = group_K_mean * (group_std_ma / group_mean_ma)

    x_pos = np.arange(len(group_keys))
    bar_colors = [colors[k[0]] for k in group_keys]
    ax2.bar(x_pos, group_K_mean, yerr=group_K_std,
            color=bar_colors, edgecolor='k', capsize=6, linewidth=0.7)

    ax2.axhline(K_fit, color='k', linestyle='--', linewidth=2,
                label=f'overall mean={K_fit:.2f} Nm/A')
    ax2.axhline(K_fit + K_std, color='gray', linestyle=':', linewidth=1)
    ax2.axhline(K_fit - K_std, color='gray', linestyle=':', linewidth=1,
                label=f'±1σ={K_std:.2f}')
    ax2.axhline(TORQUE_CONSTANT_DATASHEET, color='r', linestyle=':', linewidth=2,
                label=f'datasheet={TORQUE_CONSTANT_DATASHEET} Nm/A')

    xlabels = [f'{arm_labels[k[0]]}\n{k[1]}kg' for k in group_keys]
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(xlabels, fontsize=8)
    ax2.set_ylabel('Torque constant (Nm/A)', fontsize=11)
    ax2.set_title('K by (Arm, Weight) Group')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    from matplotlib.patches import Patch
    color_patches = [Patch(color=colors[a], label=arm_labels[a]) for a in sorted(colors)]
    ax2.legend(handles=color_patches + ax2.get_legend_handles_labels()[0], fontsize=8)

    plt.tight_layout()
    plt.savefig('/home/shashwat/awm_transformer/torque_calibration.png', dpi=150, bbox_inches='tight')
    print(f"\nSummary:")
    print(f"  Calibrated K : {K_fit:.4f} ± {K_std:.4f} Nm/A  (using 2.69 mA/LSB)")
    print(f"  Datasheet K  : {TORQUE_CONSTANT_DATASHEET} Nm/A")
    print(f"\n  Per group (arm, weight) → K mean ± std:")
    for k, km, ks in zip(group_keys, group_K_mean, group_K_std):
        print(f"    arm={k[0]}m  {k[1]}kg : {km:.3f} ± {ks:.3f} Nm/A")
    print(f"\n  Saved: torque_calibration.png")
    plt.show()

if __name__ == "__main__":
    main()
