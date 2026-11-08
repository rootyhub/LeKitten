"""Sim Arm: LeRobot joint names and degrees on the outside, MuJoCo radians inside."""
from __future__ import annotations

from lekitten.interfaces import JOINTS, Arm
from lekitten.joints import JointMap

from .world import SimWorld


class SimArm(Arm):
    def __init__(self, world: SimWorld, joint_map: dict):
        self.world = world
        self.jm = JointMap(world.model, joint_map)
        self.connected = False

    def limits_deg(self) -> dict[str, tuple[float, float]]:
        return self.jm.limits_deg()

    def connect(self) -> None:
        d = self.world.data
        for j in JOINTS:  # hold the current pose
            d.ctrl[self.jm.act[j]] = d.qpos[self.jm.qpos[j]]
        self.connected = True

    def read_joints(self) -> dict[str, float]:
        d = self.world.data
        return {j: self.jm.from_rad(j, d.qpos[self.jm.qpos[j]]) for j in JOINTS}

    def send_joints(self, joints: dict[str, float]) -> None:
        if not self.connected:
            raise RuntimeError("Arm not connected")
        d = self.world.data
        for j, v in joints.items():
            d.ctrl[self.jm.act[j]] = self.jm.to_rad(j, v)

    def disconnect(self) -> None:
        self.connected = False
