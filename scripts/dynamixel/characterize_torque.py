"""Characterize XM430-W350-T torque sensing via static load tests.

Procedure:
  1. Run noise floor measurement (no load, leg horizontal)
  2. Hang a known weight at a known distance from the joint
  3. Script records current for --duration seconds and computes calibration factor

Usage:
  python characterize_torque.py --port /dev/ttyUSB0 --id 1

Requirements:
  - A known weight (e.g. water bottle — weigh it first)
  - A ruler to measure moment arm (joint center to weight attachment point)
  - Leg held horizontal during measurement
"""

import argparse
import ctypes
import glob
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from dxl_utils import set_latency_timer

ADDR_OPERATING_MODE  = 11
ADDR_TORQUE_ENABLE   = 64
ADDR_GOAL_CURRENT    = 102
ADDR_PRESENT_CURRENT = 126
CURRENT_UNIT         = 2.69   # mA per LSB (XM430-W350-T e-manual)
TORQUE_CONSTANT_NOM  = 3.83   # Nm/A calibrated 2026-04-19
PROTOCOL_VERSION     = 2.0
BAUDRATE             = 1000000
G                    = 9.81   # m/s^2
MODE_CURRENT_BASED_POSITION = 5
DEFAULT_GOAL_CURRENT = 500    # mA


def find_port() -> str:
    candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    if not candidates:
        raise RuntimeError("No USB serial ports found.")
    return sorted(candidates)[0]


def to_signed16(val: int) -> int:
    return ctypes.c_int16(val).value


def record_current(port_handler, packet_handler, motor_id: int, duration: float, hz: float = 60.0) -> np.ndarray:
    """Record Present Current samples for `duration` seconds."""
    samples = []
    period = 1.0 / hz
    n = int(duration * hz)
    print(f"  Recording {n} samples over {duration:.1f}s ...")
    for i in range(n):
        t0 = time.monotonic()
        raw, result, error = packet_handler.read2ByteTxRx(
            port_handler, motor_id, ADDR_PRESENT_CURRENT
        )
        if result == 0 and error == 0:
            samples.append(to_signed16(raw))
        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, period - elapsed))
    return np.array(samples, dtype=float)


def print_stats(samples: np.ndarray, label: str, torque_constant: float = TORQUE_CONSTANT_NOM):
    mean_ma = np.mean(samples)
    std_ma  = np.std(samples)
    mean_nm = mean_ma * CURRENT_UNIT / 1000.0 * torque_constant
    std_nm  = std_ma  * CURRENT_UNIT / 1000.0 * torque_constant
    print(f"  {label}:")
    print(f"    Current : {mean_ma:+.2f} ± {std_ma:.2f} mA")
    print(f"    Torque  : {mean_nm:+.4f} ± {std_nm:.4f} Nm  (using {torque_constant} Nm/A)")


def wait_for_enter(msg: str):
    input(f"\n  --> {msg} [press Enter when ready]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",     type=str,   default=None)
    parser.add_argument("--id",       type=int,   default=1,    help="Motor ID")
    parser.add_argument("--baud",     type=int,   default=BAUDRATE)
    parser.add_argument("--duration", type=float, default=3.0,  help="Recording duration per trial (s)")
    parser.add_argument("--hz",       type=float, default=60.0)
    parser.add_argument("--save",         type=str,   default=None, help="Save results to JSON file")
    parser.add_argument("--goal-current", type=int,   default=DEFAULT_GOAL_CURRENT, help="Current cap in mA")
    args = parser.parse_args()

    from dynamixel_sdk import PacketHandler, PortHandler, COMM_SUCCESS

    port = args.port or find_port()
    set_latency_timer(port)
    print(f"Port: {port}  |  Motor ID: {args.id}  |  Baud: {args.baud}")

    port_handler   = PortHandler(port)
    packet_handler = PacketHandler(PROTOCOL_VERSION)

    if not port_handler.openPort():
        raise RuntimeError(f"Cannot open port {port}")
    if not port_handler.setBaudRate(args.baud):
        raise RuntimeError(f"Cannot set baud rate {args.baud}")

    # Switch to mode 5 (requires torque disabled)
    packet_handler.write1ByteTxRx(port_handler, args.id, ADDR_TORQUE_ENABLE, 0)
    packet_handler.write1ByteTxRx(port_handler, args.id, ADDR_OPERATING_MODE, MODE_CURRENT_BASED_POSITION)
    mode_rb, r, e = packet_handler.read1ByteTxRx(port_handler, args.id, ADDR_OPERATING_MODE)
    if mode_rb != MODE_CURRENT_BASED_POSITION:
        raise RuntimeError(f"Failed to set mode 5 (readback: {mode_rb})")
    print(f"Operating mode: 5 (current-based position control)")

    packet_handler.write2ByteTxRx(port_handler, args.id, ADDR_GOAL_CURRENT, args.goal_current)
    packet_handler.write1ByteTxRx(port_handler, args.id, ADDR_TORQUE_ENABLE, 1)
    print(f"Torque enabled on ID {args.id} | Goal current: {args.goal_current} mA "
          f"({args.goal_current * CURRENT_UNIT / 1000 * TORQUE_CONSTANT_NOM:.3f} Nm cap)\n")

    results = {"motor_id": args.id, "timestamp": datetime.now().isoformat(), "trials": []}

    try:
        # ── Step 1: Noise floor ──────────────────────────────────────────────
        print("=" * 55)
        print("STEP 1: Noise floor (no load)")
        print("=" * 55)
        wait_for_enter("Hold the leg horizontal with NO weight attached")
        noise_samples = record_current(port_handler, packet_handler, args.id, args.duration, args.hz)
        print_stats(noise_samples, "Noise floor")
        offset_ma = np.mean(noise_samples)
        results["noise_floor_mA"] = float(np.mean(noise_samples))
        results["noise_std_mA"]   = float(np.std(noise_samples))
        print(f"  Offset to subtract from future readings: {offset_ma:.2f} mA")

        # ── Step 2: Load trials ──────────────────────────────────────────────
        print("\n" + "=" * 55)
        print("STEP 2: Load trials")
        print("=" * 55)
        print("  You will be prompted to enter weight (kg) and arm (m)")
        print("  for each trial. Enter 'done' when finished.\n")

        trial_num = 1
        calibration_factors = []

        while True:
            print(f"── Trial {trial_num} ──────────────────────────────────────")
            weight_str = input("  Weight in kg (or 'done'): ").strip()
            if weight_str.lower() == "done":
                break
            arm_str = input("  Moment arm in m (joint center to weight): ").strip()

            try:
                weight_kg = float(weight_str)
                arm_m     = float(arm_str)
            except ValueError:
                print("  Invalid input, skipping.")
                continue

            theoretical_nm = weight_kg * G * arm_m
            print(f"  Theoretical torque: {theoretical_nm:.4f} Nm")

            wait_for_enter(f"Hang {weight_kg} kg at {arm_m} m, leg horizontal, stable")
            samples = record_current(port_handler, packet_handler, args.id, args.duration, args.hz)

            corrected = samples - offset_ma
            mean_ma   = np.mean(corrected)
            std_ma    = np.std(corrected)
            mean_a    = mean_ma * CURRENT_UNIT / 1000.0

            if abs(mean_a) > 1e-6:
                measured_constant = theoretical_nm / mean_a
            else:
                measured_constant = float("nan")

            print_stats(corrected, "Measured (offset-corrected)")
            print(f"  Theoretical      : {theoretical_nm:.4f} Nm")
            if not np.isnan(measured_constant):
                print(f"  Torque constant  : {measured_constant:.4f} Nm/A  (nominal: {TORQUE_CONSTANT_NOM})")
                print(f"  Error            : {(measured_constant - TORQUE_CONSTANT_NOM) / TORQUE_CONSTANT_NOM * 100:.1f}%")
                calibration_factors.append(measured_constant)
            else:
                print("  WARNING: near-zero current — check weight/arm setup")

            results["trials"].append({
                "trial":              trial_num,
                "weight_kg":          weight_kg,
                "arm_m":              arm_m,
                "theoretical_nm":     theoretical_nm,
                "mean_current_mA":    float(mean_ma),
                "std_current_mA":     float(std_ma),
                "torque_constant":    float(measured_constant),
            })
            trial_num += 1

        # ── Summary ─────────────────────────────────────────────────────────
        if calibration_factors:
            print("\n" + "=" * 55)
            print("SUMMARY")
            print("=" * 55)
            cal_arr = np.array(calibration_factors)
            print(f"  Torque constant across {len(cal_arr)} trial(s):")
            print(f"    Mean  : {np.mean(cal_arr):.4f} Nm/A")
            print(f"    Std   : {np.std(cal_arr):.4f} Nm/A")
            print(f"    Nominal (used in sim): {TORQUE_CONSTANT_NOM} Nm/A")
            print(f"    Recommendation: use {np.mean(cal_arr):.4f} Nm/A in read_torque.py")
            results["calibrated_constant_NmA"] = float(np.mean(cal_arr))
            results["calibrated_constant_std"]  = float(np.std(cal_arr))

        # ── Save ─────────────────────────────────────────────────────────────
        save_path = args.save or f"torque_cal_{args.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path(save_path).write_text(json.dumps(results, indent=2))
        print(f"\n  Results saved to: {save_path}")

    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        packet_handler.write1ByteTxRx(port_handler, args.id, ADDR_TORQUE_ENABLE, 0)
        port_handler.closePort()


if __name__ == "__main__":
    main()
