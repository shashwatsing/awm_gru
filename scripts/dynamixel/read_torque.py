"""Read Present Current from AWM leg motors (XM430-W350-T, TTL bus) and convert to torque.

Install: pip install dynamixel-sdk

Usage:
  python read_torque.py
  python read_torque.py --port /dev/ttyUSB0 --ids 1 2 3 4
"""

import argparse
import ctypes
import glob
import time

ADDR_TORQUE_ENABLE   = 64   # 1 byte, 0=off 1=on
ADDR_PRESENT_CURRENT = 126  # 2 bytes, signed, unit = 2.69 mA/LSB
CURRENT_UNIT    = 2.69      # mA per LSB (XM430-W350-T e-manual)
TORQUE_CONSTANT = 3.83      # Nm/A — calibrated 2026-04-19
PROTOCOL_VERSION = 2.0
BAUDRATE = 1000000


def find_port() -> str:
    candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    if not candidates:
        raise RuntimeError("No USB serial ports found. Check U2D2 connection.")
    return sorted(candidates)[0]


def to_signed16(val: int) -> int:
    return ctypes.c_int16(val).value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--ids", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--baud", type=int, default=BAUDRATE)
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--enable-torque", action="store_true", help="Enable torque so motor holds position")
    args = parser.parse_args()

    from dynamixel_sdk import PacketHandler, PortHandler, COMM_SUCCESS

    port = args.port or find_port()
    print(f"Port: {port}  |  IDs: {args.ids}  |  Baud: {args.baud}  |  {args.hz}Hz\n")

    port_handler = PortHandler(port)
    packet_handler = PacketHandler(PROTOCOL_VERSION)

    if not port_handler.openPort():
        raise RuntimeError(f"Cannot open port {port}")
    if not port_handler.setBaudRate(args.baud):
        raise RuntimeError(f"Cannot set baud rate {args.baud}")

    if args.enable_torque:
        for motor_id in args.ids:
            packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_TORQUE_ENABLE, 1)
        print(f"Torque enabled on IDs: {args.ids}")

    print(f"{'Time':>8}  {'Hz':>8}  {'ID':>4}  {'mA':>10}  {'Nm':>10}  {'Status':>10}")

    period = 1.0 / args.hz
    last_t = time.monotonic()
    try:
        while True:
            t0 = time.monotonic()
            measured_hz = 1.0 / (t0 - last_t) if (t0 - last_t) > 0 else 0.0
            last_t = t0
            rows = []
            for motor_id in args.ids:
                raw, result, error = packet_handler.read2ByteTxRx(
                    port_handler, motor_id, ADDR_PRESENT_CURRENT
                )
                if result != COMM_SUCCESS:
                    rows.append(f"{motor_id:>4}  {'---':>10}  {'---':>10}  {packet_handler.getTxRxResult(result):>10}")
                elif error != 0:
                    rows.append(f"{motor_id:>4}  {'---':>10}  {'---':>10}  {packet_handler.getRxPacketError(error):>10}")
                else:
                    ma = to_signed16(raw)
                    nm = ma * CURRENT_UNIT / 1000.0 * TORQUE_CONSTANT
                    rows.append(f"{motor_id:>4}  {ma:>10.1f}  {nm:>10.4f}  {'OK':>10}")

            ts = time.strftime("%H:%M:%S")
            for row in rows:
                print(f"{ts:>8}  {measured_hz:>6.1f}Hz  {row}")

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, period - elapsed))

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if args.enable_torque:
            for motor_id in args.ids:
                packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_TORQUE_ENABLE, 0)
        port_handler.closePort()


if __name__ == "__main__":
    main()
