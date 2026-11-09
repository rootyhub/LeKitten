"""Real Laser: an LCUS-1 USB relay (CH340 serial chip) switching the laser's USB power.

Relay commands (9600 baud): ON = A0 01 01 A2, OFF = A0 01 00 A1.
The relay is switched OFF when this object is created and again on close, whatever happens.
"""
from __future__ import annotations

import atexit

from lekitten.interfaces import Laser

ON = bytes([0xA0, 0x01, 0x01, 0xA2])
OFF = bytes([0xA0, 0x01, 0x00, 0xA1])
CH340_VID = 0x1A86


def find_relay_port(arm_port: str | None = None) -> str | None:
    """The relay's port, if exactly one CH340-style port is plugged in besides the arm's.
    (The arm's driver board can use the same chip, so ambiguity is an error, not a guess.)"""
    from serial.tools import list_ports
    found = [p.device for p in list_ports.comports()
             if (p.vid == CH340_VID or "CH340" in (p.description or "") or "usbserial" in p.device)
             and p.device != arm_port and p.device.replace("/cu.", "/tty.") != (arm_port or "").replace("/cu.", "/tty.")]
    if len(found) > 1:
        raise SystemExit(f"Several possible relay ports {found}. Set real.relay_port in config.yaml.")
    return found[0] if found else None


class RelayLaser(Laser):
    def __init__(self, port: str | None = None, dry_run: bool = False, arm_port: str | None = None):
        self.dry_run = dry_run
        self._on = False
        self.ser = None
        if dry_run:
            print(f"[dry-run] would open relay on {port or 'auto-detected port'}, then send OFF")
            return
        import serial
        port = port or find_relay_port(arm_port)
        if not port:
            raise SystemExit("No USB relay found. Plug in the LCUS-1 relay, or set real.relay_port in config.yaml.")
        if arm_port and port == arm_port:
            raise SystemExit(f"Relay port {port} is the arm's port. Set real.relay_port in config.yaml.")
        self.ser = serial.Serial(port, 9600, timeout=1)
        self._write(OFF)
        atexit.register(self.close)

    def _write(self, cmd: bytes) -> None:
        if self.dry_run:
            return
        self.ser.write(cmd)
        self.ser.flush()

    def on(self) -> None:
        if not self._on and self.dry_run:
            print("[dry-run] laser ON  (relay A0 01 01 A2)")
        self._write(ON)
        self._on = True

    def off(self) -> None:
        if self._on and self.dry_run:
            print("[dry-run] laser OFF (relay A0 01 00 A1)")
        self._write(OFF)
        self._on = False

    @property
    def is_on(self) -> bool:
        return self._on

    def close(self) -> None:
        if self.ser is not None:
            try:
                self.off()
            finally:
                self.ser.close()
                self.ser = None
