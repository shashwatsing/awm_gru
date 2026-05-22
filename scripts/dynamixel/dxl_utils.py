"""Shared utilities for Dynamixel scripts."""

import os
import glob


def set_latency_timer(port: str, ms: int = 1) -> bool:
    """Set FTDI USB latency timer for the given port.

    Without this, the FTDI chip buffers data for 16ms, limiting read rate to ~31Hz.
    Setting to 1ms allows 60Hz+ with two reads per loop.

    For a permanent fix (no sudo needed), install the udev rule:
      sudo cp 99-dynamixel.rules /etc/udev/rules.d/
      sudo udevadm control --reload-rules
    """
    dev = os.path.basename(port)
    sysfs = f"/sys/bus/usb-serial/devices/{dev}/latency_timer"
    try:
        with open(sysfs, "w") as f:
            f.write(str(ms))
        print(f"Latency timer set to {ms}ms on {dev}")
        return True
    except PermissionError:
        print(f"WARNING: Cannot set latency timer (permission denied).")
        print(f"  Run once: echo {ms} | sudo tee {sysfs}")
        print(f"  Or install udev rule for permanent fix (see 99-dynamixel.rules)")
        return False
    except FileNotFoundError:
        print(f"WARNING: Latency timer sysfs not found for {dev} — non-FTDI device?")
        return False
