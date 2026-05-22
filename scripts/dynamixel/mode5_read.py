"""Switch leg motor to current-based position control (mode 5) and read live torque.

In mode 5, Goal Current acts as a torque cap — Present Current directly
reflects motor effort, giving a much cleaner torque signal than mode 3.

Usage:
  python mode5_read.py --port /dev/ttyUSB0 --id 1
  python mode5_read.py --port /dev/ttyUSB0 --id 1 --goal-current 500
"""

import argparse
import ctypes
import glob
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from dxl_utils import set_latency_timer

ADDR_OPERATING_MODE   = 11   # 1 byte, EEPROM (write only when torque disabled)
ADDR_TORQUE_ENABLE    = 64   # 1 byte
ADDR_GOAL_CURRENT     = 102  # 2 bytes, signed, unit = 2.69 mA/LSB
ADDR_PRESENT_CURRENT  = 126  # 2 bytes, signed, unit = 2.69 mA/LSB
ADDR_PRESENT_POSITION = 132  # 4 bytes, signed, unit = 1 tick

MODE_CURRENT_BASED_POSITION = 5
CURRENT_UNIT    = 2.69   # mA per LSB (XM430-W350-T e-manual)
TORQUE_CONSTANT = 3.83   # Nm/A — calibrated 2026-04-19 (18 trials, 3 arm lengths)
TICKS_PER_REV   = 4096
PROTOCOL_VERSION = 2.0
BAUDRATE         = 1000000


def find_port() -> str:
    candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    if not candidates:
        raise RuntimeError("No USB serial ports found.")
    return sorted(candidates)[0]


def to_signed16(val: int) -> int:
    return ctypes.c_int16(val).value


def to_signed32(val: int) -> int:
    return ctypes.c_int32(val).value


def ticks_to_rad(ticks: int) -> float:
    import math
    return ticks * 2 * math.pi / TICKS_PER_REV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",         type=str, default=None)
    parser.add_argument("--id",           type=int, default=1)
    parser.add_argument("--baud",         type=int, default=BAUDRATE)
    parser.add_argument("--goal-current", type=int, default=500,
                        help="Max current in mA (torque cap). Default 500 mA (~0.90 Nm)")
    parser.add_argument("--hz",           type=float, default=60.0)
    args = parser.parse_args()

    from dynamixel_sdk import PacketHandler, PortHandler, COMM_SUCCESS

    port = args.port or find_port()
    print(f"Port: {port}  |  ID: {args.id}  |  Goal current: {args.goal_current} mA "
          f"({args.goal_current * CURRENT_UNIT / 1000 * TORQUE_CONSTANT:.3f} Nm cap)")
    set_latency_timer(port)

    port_handler   = PortHandler(port)
    packet_handler = PacketHandler(PROTOCOL_VERSION)

    if not port_handler.openPort():
        raise RuntimeError(f"Cannot open port {port}")
    if not port_handler.setBaudRate(args.baud):
        raise RuntimeError(f"Cannot set baud rate {args.baud}")

    # Read current mode
    mode, r, e = packet_handler.read1ByteTxRx(port_handler, args.id, ADDR_OPERATING_MODE)
    print(f"Current operating mode: {mode}")

    # Switch to mode 5 if needed (requires torque disabled)
    packet_handler.write1ByteTxRx(port_handler, args.id, ADDR_TORQUE_ENABLE, 0)
    if mode != MODE_CURRENT_BASED_POSITION:
        packet_handler.write1ByteTxRx(port_handler, args.id, ADDR_OPERATING_MODE, MODE_CURRENT_BASED_POSITION)
        mode_rb, r, e = packet_handler.read1ByteTxRx(port_handler, args.id, ADDR_OPERATING_MODE)
        if mode_rb != MODE_CURRENT_BASED_POSITION:
            raise RuntimeError(f"Failed to set mode 5 (readback: {mode_rb})")
        print(f"Switched to mode 5 (current-based position control)")
    else:
        print("Already in mode 5")

    # Set goal current (torque cap)
    packet_handler.write2ByteTxRx(port_handler, args.id, ADDR_GOAL_CURRENT, args.goal_current)

    # Hold current position — read position and set as goal
    pos_raw, r, e = packet_handler.read4ByteTxRx(port_handler, args.id, ADDR_PRESENT_POSITION)
    hold_pos = to_signed32(pos_raw)
    # Goal Position address = 116, 4 bytes
    packet_handler.write4ByteTxRx(port_handler, args.id, 116, hold_pos)

    # Enable torque
    packet_handler.write1ByteTxRx(port_handler, args.id, ADDR_TORQUE_ENABLE, 1)
    print(f"Torque enabled. Holding position {hold_pos} ticks ({ticks_to_rad(hold_pos):.4f} rad)\n")

    print(f"{'Time':>8}  {'Hz':>7}  {'Pos(rad)':>10}  {'mA':>8}  {'Nm':>8}  {'Cap(mA)':>8}")
    print("-" * 62)

    period = 1.0 / args.hz
    last_t = time.monotonic()
    try:
        while True:
            t0 = time.monotonic()
            hz = 1.0 / (t0 - last_t) if (t0 - last_t) > 0 else 0.0
            last_t = t0

            pos_raw, rp, ep = packet_handler.read4ByteTxRx(port_handler, args.id, ADDR_PRESENT_POSITION)
            cur_raw, rc, ec = packet_handler.read2ByteTxRx(port_handler, args.id, ADDR_PRESENT_CURRENT)

            pos_ticks = to_signed32(pos_raw)
            pos_rad   = ticks_to_rad(pos_ticks)
            ma        = to_signed16(cur_raw)
            nm        = ma * CURRENT_UNIT / 1000.0 * TORQUE_CONSTANT
            at_cap    = abs(ma) >= args.goal_current * 0.95

            cap_str = f"{'!CAP!':>8}" if at_cap else f"{args.goal_current:>8}"
            ts = time.strftime("%H:%M:%S")
            print(f"{ts:>8}  {hz:>6.1f}Hz  {pos_rad:>+10.4f}  {ma:>+8.1f}  {nm:>+8.4f}  {cap_str}")

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, period - elapsed))

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        packet_handler.write1ByteTxRx(port_handler, args.id, ADDR_TORQUE_ENABLE, 0)
        port_handler.closePort()


if __name__ == "__main__":
    main()
