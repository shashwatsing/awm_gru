"""Find and set Min/Max Position Limits for AWM leg motors.

Procedure:
  1. Torque is disabled so you can manually move the leg
  2. Move leg to each extreme, press Enter to record the position
  3. Script writes the limits to EEPROM (persists across power cycles)

Usage:
  python set_position_limits.py --port /dev/ttyUSB0 --id 1
"""

import argparse
import ctypes
import glob
import time

ADDR_TORQUE_ENABLE    = 64
ADDR_PRESENT_POSITION = 132  # 4 bytes, signed
ADDR_MAX_POSITION     = 48   # 4 bytes — EEPROM, requires torque disabled
ADDR_MIN_POSITION     = 52   # 4 bytes — EEPROM, requires torque disabled

PROTOCOL_VERSION = 2.0
BAUDRATE         = 1000000
TICKS_PER_REV    = 4096


def find_port() -> str:
    candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    if not candidates:
        raise RuntimeError("No USB serial ports found.")
    return sorted(candidates)[0]


def read_position(port_handler, packet_handler, motor_id: int) -> int:
    raw, r, e = packet_handler.read4ByteTxRx(port_handler, motor_id, ADDR_PRESENT_POSITION)
    return ctypes.c_int32(raw).value


def ticks_to_deg(ticks: int) -> float:
    return ticks * 360.0 / TICKS_PER_REV


def ticks_to_rad(ticks: int) -> float:
    import math
    return ticks * 2 * math.pi / TICKS_PER_REV


def live_position(port_handler, packet_handler, motor_id: int, msg: str) -> int:
    """Show live position readout, return ticks when user presses Enter."""
    import threading
    print(f"\n  --> {msg}")
    print("  Move leg to position, then press Enter to record.\n")

    stop = threading.Event()
    last = [0]

    def stream():
        while not stop.is_set():
            t0 = time.monotonic()
            pos = read_position(port_handler, packet_handler, motor_id)
            last[0] = pos
            print(f"\r    Position: {pos:+7d} ticks  {ticks_to_deg(pos):+8.2f} deg  {ticks_to_rad(pos):+6.4f} rad  ",
                  end="", flush=True)
            time.sleep(max(0.0, 0.05 - (time.monotonic() - t0)))

    t = threading.Thread(target=stream, daemon=True)
    t.start()
    input()
    stop.set()
    t.join()
    final = read_position(port_handler, packet_handler, motor_id)
    print(f"\n  Recorded: {final:+d} ticks  ({ticks_to_deg(final):+.2f} deg  /  {ticks_to_rad(final):+.4f} rad)")
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--id",   type=int, default=1)
    parser.add_argument("--baud", type=int, default=BAUDRATE)
    parser.add_argument("--dry-run", action="store_true", help="Don't write to EEPROM, just show values")
    args = parser.parse_args()

    from dynamixel_sdk import PacketHandler, PortHandler

    port = args.port or find_port()
    print(f"Port: {port}  |  Motor ID: {args.id}  |  Baud: {args.baud}")

    port_handler   = PortHandler(port)
    packet_handler = PacketHandler(PROTOCOL_VERSION)

    if not port_handler.openPort():
        raise RuntimeError(f"Cannot open port {port}")
    if not port_handler.setBaudRate(args.baud):
        raise RuntimeError(f"Cannot set baud rate {args.baud}")

    # Ensure torque is disabled (required to write EEPROM)
    packet_handler.write1ByteTxRx(port_handler, args.id, ADDR_TORQUE_ENABLE, 0)
    print("Torque disabled — move leg freely.\n")

    # Read current limits
    max_raw, r, e = packet_handler.read4ByteTxRx(port_handler, args.id, ADDR_MAX_POSITION)
    min_raw, r, e = packet_handler.read4ByteTxRx(port_handler, args.id, ADDR_MIN_POSITION)
    max_cur = ctypes.c_int32(max_raw).value
    min_cur = ctypes.c_int32(min_raw).value
    print(f"Current limits:")
    print(f"  Max: {max_cur:+d} ticks  ({ticks_to_deg(max_cur):+.2f} deg)")
    print(f"  Min: {min_cur:+d} ticks  ({ticks_to_deg(min_cur):+.2f} deg)")

    # Record max (most extended)
    max_ticks = live_position(port_handler, packet_handler, args.id,
                              "Move leg to MAX safe position (most extended, away from wheel)")

    # Record min (most closed)
    min_ticks = live_position(port_handler, packet_handler, args.id,
                              "Move leg to MIN safe position (most closed, away from wheel)")

    print(f"\n  Summary:")
    print(f"    Max position: {max_ticks:+d} ticks  ({ticks_to_deg(max_ticks):+.2f} deg)")
    print(f"    Min position: {min_ticks:+d} ticks  ({ticks_to_deg(min_ticks):+.2f} deg)")

    if max_ticks <= min_ticks:
        print("\n  WARNING: Max <= Min — did you set them in the right order? Aborting write.")
        port_handler.closePort()
        return

    if args.dry_run:
        print("\n  Dry run — not writing to EEPROM.")
    else:
        confirm = input("\n  Write these limits to EEPROM? (yes/no): ").strip().lower()
        if confirm == "yes":
            r1, e1 = packet_handler.write4ByteTxRx(port_handler, args.id, ADDR_MAX_POSITION, max_ticks)
            r2, e2 = packet_handler.write4ByteTxRx(port_handler, args.id, ADDR_MIN_POSITION, min_ticks)
            if r1 == 0 and r2 == 0:
                print("  Limits written successfully.")
            else:
                print(f"  Write error — max result: {r1}, min result: {r2}")
        else:
            print("  Aborted.")

    port_handler.closePort()


if __name__ == "__main__":
    main()
