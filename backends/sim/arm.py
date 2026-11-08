"""Sim Arm: LeRobot joint names and degrees on the outside, MuJoCo radians inside."""
from __future__ import annotations

import math

import mujoco

from lekitten.interfaces import JOINTS, Arm

from .world import SimWorld


def _mj_name(joint: str) -> str:
    return joint.removesuffix(".pos")  # "shoulder_pan.pos" -> "shoulder_pan"


class SimArm(Arm):
    def __init__(self, world: SimWorld, joint_map: dict):
        self.world = world
        m = world.model
        self._map = joint_map
        self._qpos, self._act, self._range = {}, {}, {}
        for j in JOINTS:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, _mj_name(j))
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, _mj_name(j))
            if jid < 0 or aid < 0:
                raise ValueError(f"Joint/actuator {_mj_name(j)!r} not in the MuJoCo model")
            self._qpos[j] = m.jnt_qposadr[jid]
            self._act[j] = aid
            self._range[j] = tuple(m.actuator_ctrlrange[aid])
        self.connected = False

    # --- unit conversion (the one place sim units appear) -------------------
    def to_sim(self, joint: str, value: float) -> float:
        lo, hi = self._range[joint]
        if joint == "gripper.pos":
            # LeRobot: 0 = open, 100 = closed. Model: hi = open, lo = closed.
            pct = min(max(value, 0.0), 100.0) / 100.0
            return hi - pct * (hi - lo)
        jm = self._map[joint]
        rad = jm["sign"] * math.radians(value) + jm.get("offset_rad", 0.0)
        return min(max(rad, lo), hi)

    def from_sim(self, joint: str, rad: float) -> float:
        lo, hi = self._range[joint]
        if joint == "gripper.pos":
            return 100.0 * (hi - rad) / (hi - lo)
        jm = self._map[joint]
        return math.degrees((rad - jm.get("offset_rad", 0.0)) / jm["sign"])

    def limits_deg(self) -> dict[str, tuple[float, float]]:
        out = {}
        for j in JOINTS:
            a, b = (self.from_sim(j, r) for r in self._range[j])
            out[j] = (min(a, b), max(a, b))
        return out

    # --- Arm interface ----------------------------------------------------
    def connect(self) -> None:
        d = self.world.data
        for j in JOINTS:  # hold the current pose
            d.ctrl[self._act[j]] = d.qpos[self._qpos[j]]
        self.connected = True

    def read_joints(self) -> dict[str, float]:
        d = self.world.data
        return {j: self.from_sim(j, d.qpos[self._qpos[j]]) for j in JOINTS}

    def send_joints(self, joints: dict[str, float]) -> None:
        if not self.connected:
            raise RuntimeError("Arm not connected")
        d = self.world.data
        for j, v in joints.items():
            d.ctrl[self._act[j]] = self.to_sim(j, v)

    def disconnect(self) -> None:
        self.connected = False
