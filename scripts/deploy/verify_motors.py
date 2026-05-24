"""
Motor verification script — wiggle one motor at a time to confirm ID↔joint mapping.

Wheels: spins each wheel briefly at low speed, one at a time.
Legs:   extends each leg slightly, then returns to closed, one at a time.

Expected mapping:
  Wheel ID 0 = B_R,  ID 1 = F_R,  ID 2 = F_L,  ID 3 = B_L
  Leg   ID 0 = B_R,  ID 1 = F_R,  ID 2 = F_L,  ID 3 = B_L

Usage:
  python verify_motors.py          # verify all motors
  python verify_motors.py --wheels # wheels only
  python verify_motors.py --legs   # legs only
"""

import argparse
import struct
import time

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

WHEEL_PORT = "/dev/ttyUSB1"
LEG_PORT   = "/dev/ttyUSB0"
BAUDRATE   = 1_000_000
PROTOCOL   = 2.0

WHEEL_IDS = [0, 1, 2, 3]
LEG_IDS   = [0, 1, 2, 3]

WHEEL_LABELS = {0: "B_R", 1: "F_R", 2: "F_L", 3: "B_L"}
LEG_LABELS   = {0: "B_R", 1: "F_R", 2: "F_L", 3: "B_L"}

ADDR_TORQUE_ENABLE   = 64
ADDR_GOAL_VELOCITY   = 104   # wheels
ADDR_GOAL_POSITION   = 116   # legs
ADDR_PRESENT_POSITION = 132

# Leg tick constants
# Calibrated 2026-05-24 — keyed by DXL ID
LEG_CLOSED = {0: 2048, 1: 2033, 2: 1579, 3:  617}
LEG_OPEN   = {0:    2, 1:    0, 2: 3612, 3: 2678}

WHEEL_SPIN_LSB = 50   # ~0.5 rad/s — slow enough to see, safe
LEG_WIGGLE_TICKS = 200  # small nudge toward open


def pack4(val: int) -> list:
    return list(struct.pack("<i", val))


def pack1(val: int) -> list:
    return [val]


class Bus:
    def __init__(self, port: str, name: str):
        self.ph   = PortHandler(port)
        self.pkt  = PacketHandler(PROTOCOL)
        self.name = name
        if not self.ph.openPort():
            raise RuntimeError(f"Cannot open {name} port: {port}")
        self.ph.setBaudRate(BAUDRATE)
        print(f"Opened {name} bus on {port}")

    def scan(self, ids: list[int]) -> list[int]:
        found = []
        for id_ in ids:
            _, result, _ = self.pkt.ping(self.ph, id_)
            if result == COMM_SUCCESS:
                found.append(id_)
        return found

    def write1(self, id_: int, addr: int, val: int):
        self.pkt.write1ByteTxRx(self.ph, id_, addr, val)

    def write4(self, id_: int, addr: int, val: int):
        self.pkt.write4ByteTxRx(self.ph, id_, addr, val)

    def read4(self, id_: int, addr: int) -> int:
        raw, result, _ = self.pkt.read4ByteTxRx(self.ph, id_, addr)
        if result != COMM_SUCCESS:
            return -1
        val = struct.unpack("<i", struct.pack("<I", raw & 0xFFFFFFFF))[0]
        return val

    def torque(self, id_: int, on: bool):
        self.write1(id_, ADDR_TORQUE_ENABLE, 1 if on else 0)

    def close(self):
        self.ph.closePort()


def verify_wheels(bus: Bus):
    print("\n" + "="*50)
    print("WHEEL VERIFICATION")
    print("="*50)

    found = bus.scan(WHEEL_IDS)
    print(f"Found wheel IDs: {found}")
    missing = [i for i in WHEEL_IDS if i not in found]
    if missing:
        print(f"  WARNING: missing IDs {missing}")

    for id_ in found:
        label = WHEEL_LABELS[id_]
        input(f"\n  Press Enter to spin wheel ID {id_} — expected: {label} ...")
        bus.torque(id_, True)

        # Spin forward
        bus.write4(id_, ADDR_GOAL_VELOCITY, WHEEL_SPIN_LSB)
        time.sleep(1.0)

        # Stop
        bus.write4(id_, ADDR_GOAL_VELOCITY, 0)
        bus.torque(id_, False)

        answer = input(f"  Which wheel spun? (expected {label}): ").strip().upper()
        status = "✓" if answer == label else f"✗  (got {answer}, expected {label})"
        print(f"  ID {id_} → {status}")


def verify_legs(bus: Bus):
    print("\n" + "="*50)
    print("LEG VERIFICATION")
    print("="*50)

    found = bus.scan(LEG_IDS)
    print(f"Found leg IDs: {found}")
    missing = [i for i in LEG_IDS if i not in found]
    if missing:
        print(f"  WARNING: missing IDs {missing}")

    # Read current positions
    print("\nCurrent leg positions (ticks):")
    for id_ in found:
        ticks = bus.read4(id_, ADDR_PRESENT_POSITION)
        label = LEG_LABELS[id_]
        closed = LEG_CLOSED[id_]
        status = "closed ✓" if abs(ticks - closed) < 100 else f"NOT at closed ({closed})"
        print(f"  ID {id_} ({label}): {ticks} ticks — {status}")

    for id_ in found:
        label = LEG_LABELS[id_]
        closed_pos = LEG_CLOSED[id_]
        open_dir   = LEG_OPEN[id_]
        wiggle_pos = closed_pos + (open_dir - closed_pos) // 5  # 20% open

        input(f"\n  Press Enter to wiggle leg ID {id_} — expected: {label} ...")
        bus.torque(id_, True)

        # Move slightly open
        bus.write4(id_, ADDR_GOAL_POSITION, wiggle_pos)
        time.sleep(1.0)

        # Return to closed
        bus.write4(id_, ADDR_GOAL_POSITION, closed_pos)
        time.sleep(1.0)
        bus.torque(id_, False)

        answer = input(f"  Which leg moved? (expected {label}): ").strip().upper()
        # Normalise input e.g. "FL" → "F_L"
        if "_" not in answer and len(answer) == 2:
            answer = answer[0] + "_" + answer[1]
        status = "✓" if answer == label else f"✗  (got {answer}, expected {label})"
        print(f"  ID {id_} → {status}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheels", action="store_true")
    parser.add_argument("--legs",   action="store_true")
    args = parser.parse_args()
    do_all = not args.wheels and not args.legs

    wheel_bus = leg_bus = None
    try:
        if do_all or args.wheels:
            wheel_bus = Bus(WHEEL_PORT, "wheel")
            verify_wheels(wheel_bus)

        if do_all or args.legs:
            leg_bus = Bus(LEG_PORT, "leg")
            verify_legs(leg_bus)

        print("\nVerification complete.")

    finally:
        if wheel_bus:
            # Ensure all wheels stopped
            for id_ in WHEEL_IDS:
                wheel_bus.write4(id_, ADDR_GOAL_VELOCITY, 0)
                wheel_bus.torque(id_, False)
            wheel_bus.close()
        if leg_bus:
            leg_bus.close()


if __name__ == "__main__":
    main()
