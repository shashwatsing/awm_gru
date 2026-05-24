"""
Dynamixel interface for AWM robot — GroupSyncRead + GroupSyncWrite at 60 Hz.

Two buses:
  WHEEL_PORT (RS-485): 4× XM430-W210-R — velocity control
  LEG_PORT   (TTL):    4× XM430-W350-T — current-based position control (mode 5)

Motor ID assignment (verified 2026-05-24):
  ID 0 = wheel_B_R / leg_B_R   (Back Right)
  ID 1 = wheel_B_L / leg_F_R   (wheel: Back Left,  leg: Front Right)
  ID 2 = wheel_F_L / leg_F_L   (Front Left)
  ID 3 = wheel_F_R / leg_B_L   (wheel: Front Right, leg: Back Left)

Sim obs ordering (arrays throughout this file are in this order):
  Wheels: [F_L, F_R, B_R, B_L]  → IDs [2, 1, 0, 3]
  Legs:   [F_L, F_R, B_L, B_R]  → IDs [2, 1, 3, 0]

Run standalone to scan and verify IDs before first policy run:
  python dxl_interface.py
"""

import math
import struct

import numpy as np
from dynamixel_sdk import (
    COMM_SUCCESS,
    GroupSyncRead,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
)

# ── Ports ──────────────────────────────────────────────────────────────────────
WHEEL_PORT = "/dev/ttyUSB1"   # RS-485, XM430-W210-R ×4
LEG_PORT   = "/dev/ttyUSB0"   # TTL,    XM430-W350-T ×4
BAUDRATE   = 1_000_000
PROTOCOL   = 2.0

# ── Motor IDs in sim obs order ────────────────────────────────────────────────
# Wheels sim obs order: [F_L, F_R, B_R, B_L]
WHEEL_IDS = [2, 3, 0, 1]
# Legs sim obs order:   [F_L, F_R, B_L, B_R]
LEG_IDS   = [2, 1, 3, 0]

# ── Register addresses (XM430, Protocol 2.0) ──────────────────────────────────
ADDR_TORQUE_ENABLE    = 64
ADDR_GOAL_VELOCITY    = 104   # 4B signed, wheels only
ADDR_GOAL_POSITION    = 116   # 4B signed, legs only
ADDR_PRESENT_CURRENT  = 126   # 2B signed, legs only
ADDR_PRESENT_VELOCITY = 128   # 4B signed, both
ADDR_PRESENT_POSITION = 132   # 4B signed, legs only

# Single contiguous read covering Current(126,2B) + Velocity(128,4B) + Position(132,4B)
# Starting at 126, length 10 covers bytes 126-135.
LEG_READ_START  = ADDR_PRESENT_CURRENT   # 126
LEG_READ_LEN    = 10                     # covers current(2) + velocity(4) + position(4)
LEG_CUR_OFFSET  = 0                      # bytes 0-1 within the block
LEG_POS_OFFSET  = 6                      # bytes 6-9 within the block

WHEEL_READ_START = ADDR_PRESENT_VELOCITY  # 128
WHEEL_READ_LEN   = 4

# ── Physical constants ─────────────────────────────────────────────────────────
WHEEL_VEL_UNIT = 0.229 * 2 * math.pi / 60   # LSB → rad/s  (0.229 RPM/LSB)

# Sign mask in sim obs order [F_L, F_R, B_R, B_L]
WHEEL_VEL_SIGN = np.array([1.0, -1.0, -1.0, 1.0])

CURRENT_UNIT    = 2.69e-3    # A/LSB
TORQUE_CONSTANT = 3.83       # Nm/A  (calibrated 2026-04-19)
TORQUE_NORM     = 2.70       # Nm — normalisation denominator

# Leg tick↔extension mapping — calibrated 2026-05-24 on physical robot
#
# sim obs idx:    0      1      2      3
# joint:         F_L    F_R    B_L    B_R
# DXL ID:         2      1      3      0
# side:          Left  Right   Left  Right
#
# Left  legs (F_L, B_L): ticks INCREASE from closed→open
# Right legs (F_R, B_R): ticks DECREASE from closed→open (open near 0)
# Open ticks for right legs clamped to 5 (measured -28/-26, negative invalid for EEPROM)
#
LEG_CLOSED_TICKS = np.array([1579, 2033,  617, 2048], dtype=np.int32)
LEG_OPEN_TICKS   = np.array([3612,    0, 2678,    2], dtype=np.int32)

# Target sim angle at open position: left=-π, right=+π
LEG_TARGET_ANGLE = np.array([-math.pi, math.pi, -math.pi, math.pi])


def _to_signed16(raw: int) -> int:
    return struct.unpack("<h", struct.pack("<H", raw & 0xFFFF))[0]


def _to_signed32(raw: int) -> int:
    return struct.unpack("<i", struct.pack("<I", raw & 0xFFFFFFFF))[0]


def _vel_lsb_to_rad(lsb: int) -> float:
    return _to_signed32(lsb) * WHEEL_VEL_UNIT


def _rad_to_vel_lsb(rad_s: float) -> int:
    return int(round(rad_s / WHEEL_VEL_UNIT))


def _ticks_to_leg_rad(ticks: int, obs_idx: int) -> float:
    """Convert DXL ticks to sim joint angle (rad).
    obs_idx: position in sim obs array (0=F_L, 1=F_R, 2=B_L, 3=B_R).
    Formula: ratio = (ticks - closed) / (open - closed), sim_rad = ratio * target_angle
    """
    closed = int(LEG_CLOSED_TICKS[obs_idx])
    open_  = int(LEG_OPEN_TICKS[obs_idx])
    ratio  = (ticks - closed) / (open_ - closed)
    return ratio * LEG_TARGET_ANGLE[obs_idx]


def _extension_to_ticks(extension: float, obs_idx: int) -> int:
    """Convert extension ∈ [0,1] to DXL ticks.
    obs_idx: position in sim obs array (0=F_L, 1=F_R, 2=B_L, 3=B_R).
    """
    ext    = float(np.clip(extension, 0.0, 1.0))
    closed = int(LEG_CLOSED_TICKS[obs_idx])
    open_  = int(LEG_OPEN_TICKS[obs_idx])
    ticks  = int(round(closed + ext * (open_ - closed)))
    return int(np.clip(ticks, min(closed, open_), max(closed, open_)))


class DxlInterface:
    def __init__(self, wheel_port: str = WHEEL_PORT, leg_port: str = LEG_PORT):
        # ── Open ports ────────────────────────────────────────────────────────
        self._wheel_ph = PortHandler(wheel_port)
        self._leg_ph   = PortHandler(leg_port)
        self._pkt      = PacketHandler(PROTOCOL)

        for ph, name in [(self._wheel_ph, "wheel"), (self._leg_ph, "leg")]:
            if not ph.openPort():
                raise RuntimeError(f"Failed to open {name} port")
            if not ph.setBaudRate(BAUDRATE):
                raise RuntimeError(f"Failed to set baud on {name} port")

        # ── Enable torque ─────────────────────────────────────────────────────
        for id_ in WHEEL_IDS:
            self._write1(self._wheel_ph, id_, ADDR_TORQUE_ENABLE, 1)
        for id_ in LEG_IDS:
            self._write1(self._leg_ph, id_, ADDR_TORQUE_ENABLE, 1)

        # ── GroupSyncRead: wheel velocities ───────────────────────────────────
        self._wheel_sr = GroupSyncRead(self._wheel_ph, self._pkt,
                                       WHEEL_READ_START, WHEEL_READ_LEN)
        for id_ in WHEEL_IDS:
            self._wheel_sr.addParam(id_)

        # ── GroupSyncRead: leg current + position (contiguous block 126-135) ──
        self._leg_sr = GroupSyncRead(self._leg_ph, self._pkt,
                                     LEG_READ_START, LEG_READ_LEN)
        for id_ in LEG_IDS:
            self._leg_sr.addParam(id_)

        # ── GroupSyncWrite: wheel goal velocity ───────────────────────────────
        self._wheel_sw = GroupSyncWrite(self._wheel_ph, self._pkt,
                                        ADDR_GOAL_VELOCITY, 4)

        # ── GroupSyncWrite: leg goal position ─────────────────────────────────
        self._leg_sw = GroupSyncWrite(self._leg_ph, self._pkt,
                                      ADDR_GOAL_POSITION, 4)

        print(f"DxlInterface ready — wheels on {wheel_port}, legs on {leg_port}")

    # ── Read ──────────────────────────────────────────────────────────────────

    def read_wheel_velocities(self) -> np.ndarray:
        """Returns wheel velocities in sim obs order [F_L, F_R, B_R, B_L]
        in rad/s with obs sign mask applied."""
        result = self._wheel_sr.txRxPacket()
        if result != COMM_SUCCESS:
            print(f"[WARN] wheel sync read failed: {self._pkt.getTxRxResult(result)}")
            return np.zeros(4)

        vels = np.zeros(4)
        for i, id_ in enumerate(WHEEL_IDS):
            raw     = self._wheel_sr.getData(id_, WHEEL_READ_START, WHEEL_READ_LEN)
            vels[i] = _vel_lsb_to_rad(raw)

        return vels * WHEEL_VEL_SIGN

    def read_leg_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (leg_positions_rad, leg_torques_norm) in sim obs order
        [F_L, F_R, B_L, B_R]."""
        result = self._leg_sr.txRxPacket()
        if result != COMM_SUCCESS:
            print(f"[WARN] leg sync read failed: {self._pkt.getTxRxResult(result)}")
            return np.zeros(4), np.zeros(4)

        positions = np.zeros(4)
        torques   = np.zeros(4)
        for i, id_ in enumerate(LEG_IDS):
            # Current (2 bytes at offset 0)
            cur_raw    = self._leg_sr.getData(id_, LEG_READ_START + LEG_CUR_OFFSET, 2)
            cur_signed = _to_signed16(cur_raw)
            torque_nm  = cur_signed * CURRENT_UNIT * TORQUE_CONSTANT
            torques[i] = float(np.clip(torque_nm / TORQUE_NORM, -2.0, 2.0))

            # Position (4 bytes at offset 6)
            pos_raw      = self._leg_sr.getData(id_, LEG_READ_START + LEG_POS_OFFSET, 4)
            ticks        = _to_signed32(pos_raw)
            positions[i] = _ticks_to_leg_rad(ticks, i)

        return positions, torques

    # ── Write ─────────────────────────────────────────────────────────────────

    def write_wheel_velocities(self, vel_rad_s: np.ndarray) -> None:
        """vel_rad_s: sim obs order [F_L, F_R, B_R, B_L] rad/s
        (obs sign convention — sign is inverted internally before writing)."""
        raw_vels = vel_rad_s / WHEEL_VEL_SIGN   # undo sign mask for raw motor direction

        self._wheel_sw.clearParam()
        for i, id_ in enumerate(WHEEL_IDS):
            lsb  = _rad_to_vel_lsb(float(raw_vels[i]))
            data = list(struct.pack("<i", lsb))
            self._wheel_sw.addParam(id_, data)
        self._wheel_sw.txPacket()

    def write_leg_positions(self, extensions: np.ndarray) -> None:
        """extensions: sim obs order [F_L, F_R, B_L, B_R] ∈ [0,1]
        (0=closed, 1=fully open)."""
        self._leg_sw.clearParam()
        for i, id_ in enumerate(LEG_IDS):
            ticks = _extension_to_ticks(float(extensions[i]), i)
            data  = list(struct.pack("<i", ticks))
            self._leg_sw.addParam(id_, data)
        self._leg_sw.txPacket()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def scan(self) -> None:
        """Scan both buses and print found motor IDs."""
        print("\n── Wheel bus (RS-485) ──")
        for id_ in range(0, 10):
            model, result, _ = self._pkt.ping(self._wheel_ph, id_)
            if result == COMM_SUCCESS:
                print(f"  wheel_{id_}: model={model}")

        print("── Leg bus (TTL) ──")
        for id_ in range(0, 10):
            model, result, _ = self._pkt.ping(self._leg_ph, id_)
            if result == COMM_SUCCESS:
                print(f"  leg_{id_}: model={model}")

    def read_leg_ticks_raw(self) -> list[int]:
        """Read raw position ticks in sim obs order [F_L, F_R, B_L, B_R].
        Use for calibration verification."""
        ticks = []
        for id_ in LEG_IDS:
            raw, result, _ = self._pkt.read4ByteTxRx(
                self._leg_ph, id_, ADDR_PRESENT_POSITION)
            ticks.append(_to_signed32(raw) if result == COMM_SUCCESS else -1)
        return ticks

    def close(self) -> None:
        for id_ in WHEEL_IDS:
            self._write1(self._wheel_ph, id_, ADDR_TORQUE_ENABLE, 0)
        for id_ in LEG_IDS:
            self._write1(self._leg_ph, id_, ADDR_TORQUE_ENABLE, 0)
        self._wheel_ph.closePort()
        self._leg_ph.closePort()

    def _write1(self, ph: PortHandler, id_: int, addr: int, val: int) -> None:
        result, _ = self._pkt.write1ByteTxRx(ph, id_, addr, val)
        if result != COMM_SUCCESS:
            print(f"[WARN] write1 id={id_} addr={addr}: {self._pkt.getTxRxResult(result)}")


# ── Standalone scan ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    dxl = DxlInterface()
    dxl.scan()

    print("\nLeg ticks in sim obs order [F_L, F_R, B_L, B_R]:")
    print("(for calibration — confirm closed≈3641 for F_L/B_L, ≈1631 for F_R/B_R)")
    ticks = dxl.read_leg_ticks_raw()
    labels = ["F_L (ID 2)", "F_R (ID 1)", "B_L (ID 3)", "B_R (ID 0)"]
    for label, id_, t in zip(labels, LEG_IDS, ticks):
        print(f"  {label}  ticks={t}")
    dxl.close()
