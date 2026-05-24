"""
Set position limits for all 4 AWM leg motors in one run.

Procedure per leg:
  1. Torque disabled — move leg manually to each extreme
  2. Press Enter to record closed position, then open position
  3. Limits written to EEPROM (persists across power cycles)

Leg IDs and expected positions:
  ID 0 (B_R, Right): closed ≈ 1631 ticks, open ≈ 3641 ticks
  ID 1 (F_R, Right): closed ≈ 1631 ticks, open ≈ 3641 ticks
  ID 2 (F_L, Left):  closed ≈ 3641 ticks, open ≈ 1631 ticks
  ID 3 (B_L, Left):  closed ≈ 3641 ticks, open ≈ 1631 ticks

Usage:
  python set_all_leg_limits.py
  python set_all_leg_limits.py --port /dev/ttyUSB1
  python set_all_leg_limits.py --dry-run   # show without writing
"""

import argparse
import ctypes
import math
import threading
import time

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

LEG_PORT  = "/dev/ttyUSB1"
BAUDRATE  = 1_000_000
PROTOCOL  = 2.0

# IDs in physical order (not sim obs order) — set limits one leg at a time
LEG_IDS    = [0, 1, 2, 3]
LEG_LABELS = {0: "B_R (Right)", 1: "F_R (Right)", 2: "F_L (Left)", 3: "B_L (Left)"}

ADDR_TORQUE_ENABLE    = 64
ADDR_PRESENT_POSITION = 132
ADDR_MAX_POSITION     = 48   # EEPROM — requires torque disabled
ADDR_MIN_POSITION     = 52   # EEPROM — requires torque disabled

TICKS_PER_REV = 4096


def ticks_to_deg(t): return t * 360.0 / TICKS_PER_REV
def ticks_to_rad(t): return t * 2 * math.pi / TICKS_PER_REV


def read_pos(ph, pkt, id_) -> int:
    raw, _, _ = pkt.read4ByteTxRx(ph, id_, ADDR_PRESENT_POSITION)
    return ctypes.c_int32(raw).value


def live_record(ph, pkt, id_, prompt: str) -> int:
    """Show live position stream, return ticks when user presses Enter."""
    print(f"\n  {prompt}")
    print("  Move leg to position, then press Enter to record.")

    stop = threading.Event()
    last = [0]

    def stream():
        while not stop.is_set():
            pos = read_pos(ph, pkt, id_)
            last[0] = pos
            print(f"\r    {pos:+7d} ticks  {ticks_to_deg(pos):+8.2f}°  {ticks_to_rad(pos):+.4f} rad  ",
                  end="", flush=True)
            time.sleep(0.05)

    t = threading.Thread(target=stream, daemon=True)
    t.start()
    input()
    stop.set()
    t.join()
    final = read_pos(ph, pkt, id_)
    print(f"\n  Recorded: {final:+d} ticks  ({ticks_to_deg(final):.2f}°  /  {ticks_to_rad(final):.4f} rad)")
    return final


def set_limits_for(ph, pkt, id_: int, dry_run: bool):
    label = LEG_LABELS[id_]
    print(f"\n{'='*55}")
    print(f"  Leg ID {id_}  —  {label}")
    print(f"{'='*55}")

    # Read existing limits
    max_raw, _, _ = pkt.read4ByteTxRx(ph, id_, ADDR_MAX_POSITION)
    min_raw, _, _ = pkt.read4ByteTxRx(ph, id_, ADDR_MIN_POSITION)
    print(f"  Current EEPROM limits:")
    print(f"    Max: {ctypes.c_int32(max_raw).value:+d} ticks")
    print(f"    Min: {ctypes.c_int32(min_raw).value:+d} ticks")

    # Disable torque so leg can be moved freely
    pkt.write1ByteTxRx(ph, id_, ADDR_TORQUE_ENABLE, 0)
    print("  Torque disabled — leg moves freely.")

    closed_ticks = live_record(ph, pkt, id_, "Move leg to CLOSED position (wheel touching ground, leg retracted)")
    open_ticks   = live_record(ph, pkt, id_, "Move leg to OPEN position (leg fully extended, max safe range)")

    # min/max in EEPROM must be min < max regardless of which is closed/open
    min_ticks = min(closed_ticks, open_ticks)
    max_ticks = max(closed_ticks, open_ticks)

    print(f"\n  Summary for ID {id_} ({label}):")
    print(f"    Closed : {closed_ticks:+d} ticks  ({ticks_to_deg(closed_ticks):.2f}°)")
    print(f"    Open   : {open_ticks:+d}   ticks  ({ticks_to_deg(open_ticks):.2f}°)")
    print(f"    EEPROM → Min: {min_ticks:+d},  Max: {max_ticks:+d},  Range: {max_ticks - min_ticks} ticks  ({ticks_to_rad(max_ticks - min_ticks):.3f} rad)")

    if max_ticks - min_ticks < 100:
        print("  WARNING: range < 100 ticks — did you move the leg enough? Skipping write.")
        return None, None

    if not dry_run:
        confirm = input("  Write to EEPROM? (yes/no): ").strip().lower()
        if confirm == "yes":
            r1, _ = pkt.write4ByteTxRx(ph, id_, ADDR_MAX_POSITION, max_ticks)
            r2, _ = pkt.write4ByteTxRx(ph, id_, ADDR_MIN_POSITION, min_ticks)
            if r1 == COMM_SUCCESS and r2 == COMM_SUCCESS:
                print("  ✓ Limits written to EEPROM.")
            else:
                print(f"  ✗ Write error — max: {r1}, min: {r2}")
        else:
            print("  Skipped.")
    else:
        print("  [dry-run] Not writing.")

    return closed_ticks, open_ticks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",    default=LEG_PORT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--id",      type=int, default=None,
                        help="Set limits for a single ID only (default: all)")
    args = parser.parse_args()

    ph  = PortHandler(args.port)
    pkt = PacketHandler(PROTOCOL)

    if not ph.openPort():
        raise RuntimeError(f"Cannot open port {args.port}")
    ph.setBaudRate(BAUDRATE)
    print(f"Opened leg bus on {args.port}")

    ids = [args.id] if args.id is not None else LEG_IDS

    # Scan first
    print("\nScanning for leg motors...")
    found = []
    for id_ in ids:
        _, result, _ = pkt.ping(ph, id_)
        status = "✓" if result == COMM_SUCCESS else "✗ NOT FOUND"
        label  = LEG_LABELS.get(id_, "?")
        print(f"  ID {id_} ({label}): {status}")
        if result == COMM_SUCCESS:
            found.append(id_)

    if not found:
        print("No motors found — check port and power.")
        ph.closePort()
        return

    results = {}
    for id_ in found:
        closed, open_ = set_limits_for(ph, pkt, id_, args.dry_run)
        if closed is not None:
            results[id_] = {"closed": closed, "open": open_}

    # Final summary
    print(f"\n{'='*55}")
    print("  FINAL SUMMARY")
    print(f"{'='*55}")
    print(f"  {'ID':<4} {'Label':<16} {'Closed':>8} {'Open':>8} {'Range':>8}")
    for id_, vals in results.items():
        rng = abs(vals['open'] - vals['closed'])
        print(f"  {id_:<4} {LEG_LABELS[id_]:<16} {vals['closed']:>8} {vals['open']:>8} {rng:>8}")

    print("\nCopy these values into dxl_interface.py:")
    print("  LEG_CLOSED_TICKS (sim obs order [F_L, F_R, B_L, B_R] = IDs [2, 1, 3, 0]):")
    obs_order = [2, 1, 3, 0]  # sim obs order → IDs
    labels_obs = ["F_L", "F_R", "B_L", "B_R"]
    for label, id_ in zip(labels_obs, obs_order):
        if id_ in results:
            print(f"    {label} (ID {id_}): closed={results[id_]['closed']}, open={results[id_]['open']}")

    ph.closePort()


if __name__ == "__main__":
    main()
