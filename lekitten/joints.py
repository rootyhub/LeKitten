"""LeRobot joint units (degrees, gripper 0..100) <-> MuJoCo radians, via joint_map in config.yaml."""
from __future__ import annotations

import math

import mujoco

from .interfaces import JOINTS


def mj_name(joint: str) -> str:
    return joint.removesuffix(".pos")  # "shoulder_pan.pos" -> "shoulder_pan"


class JointMap:
    def __init__(self, model: mujoco.MjModel, joint_map: dict):
        self._map = joint_map
        self.qpos, self.act, self.range = {}, {}, {}
        for j in JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj_name(j))
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, mj_name(j))
            if jid < 0 or aid < 0:
                raise ValueError(f"Joint/actuator {mj_name(j)!r} not in the MuJoCo model")
            self.qpos[j] = model.jnt_qposadr[jid]
            self.act[j] = aid
            self.range[j] = tuple(model.actuator_ctrlrange[aid])

    def to_rad(self, joint: str, value: float) -> float:
        lo, hi = self.range[joint]
        if joint == "gripper.pos":
            # LeRobot: 0 = open, 100 = closed. Model: hi = open, lo = closed.
            pct = min(max(value, 0.0), 100.0) / 100.0
            return hi - pct * (hi - lo)
        jm = self._map[joint]
        rad = jm["sign"] * math.radians(value) + jm.get("offset_rad", 0.0)
        return min(max(rad, lo), hi)

    def from_rad(self, joint: str, rad: float) -> float:
        lo, hi = self.range[joint]
        if joint == "gripper.pos":
            return 100.0 * (hi - rad) / (hi - lo)
        jm = self._map[joint]
        return math.degrees((rad - jm.get("offset_rad", 0.0)) / jm["sign"])

    def limits_deg(self) -> dict[str, tuple[float, float]]:
        out = {}
        for j in JOINTS:
            a, b = (self.from_rad(j, r) for r in self.range[j])
            out[j] = (min(a, b), max(a, b))
        return out
