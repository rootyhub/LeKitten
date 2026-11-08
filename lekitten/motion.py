"""Smooth moves built only on the Arm interface, so they behave the same in sim and real."""
from __future__ import annotations

import math

from .backend import Backend

HOME = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 0.0,
}


def move_to(b: Backend, target: dict[str, float], max_speed_deg_s: float = 60.0,
            rate_hz: float = 50.0) -> None:
    """Glide from the current pose to `target` without exceeding max_speed_deg_s on any joint."""
    start = b.arm.read_joints()
    span = max(abs(target[j] - start[j]) for j in target)
    duration = max(span / max_speed_deg_s, 0.2)
    steps = max(1, int(duration * rate_hz))
    for i in range(1, steps + 1):
        s = 0.5 - 0.5 * math.cos(math.pi * i / steps)  # ease in/out
        b.arm.send_joints({j: start[j] + s * (target[j] - start[j]) for j in target})
        b.wait(1.0 / rate_hz)
    b.wait(0.3)  # let it settle
