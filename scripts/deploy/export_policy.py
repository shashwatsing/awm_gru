"""
Export GRU policy from checkpoint to JIT format for deployment.
Run once on the workstation before copying to Jetson.

Usage:
    cd /home/shashwat/awm_gru
    conda run -n env_isaaclab python scripts/deploy/export_policy.py \
        logs/rsl_rl/awm_gru_proprio_torque/<timestamp>/model_best.pt

Output: <checkpoint_dir>/exported/policy.pt  (JIT, includes obs normalizer + GRU state)
        <checkpoint_dir>/exported/policy.onnx
"""

import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print("Usage: export_policy.py <path/to/model_best.pt>")
        sys.exit(1)

    checkpoint = Path(sys.argv[1]).resolve()
    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        sys.exit(1)

    export_dir = checkpoint.parent / "exported"
    print(f"Exporting from: {checkpoint}")
    print(f"Output dir:     {export_dir}")

    # play.py exports policy.pt + policy.onnx on startup before the sim loop
    subprocess.run([
        "python", "scripts/rsl_rl/play.py",
        "--task", "Template-Awm_GRU_ProprioTorque-v0",
        "--num_envs", "1",
        "--checkpoint", str(checkpoint),
        "--headless",
        "--video_length", "1",  # exit after 1 frame
    ], check=True)

    jit_path = export_dir / "policy.pt"
    if jit_path.exists():
        print(f"\nExport successful: {jit_path}")
        print("Copy to Jetson:")
        print(f"  scp {jit_path} jetson:~/awm_deploy/exported/policy.pt")
    else:
        print(f"\nERROR: expected {jit_path} but not found — check play.py output above")
        sys.exit(1)


if __name__ == "__main__":
    main()
