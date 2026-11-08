"""Sim Laser: shows a red dot where the beam first hits the floor or furniture, only while on."""
from __future__ import annotations

import mujoco

from lekitten.interfaces import Laser
from lekitten.kinematics import Beam

from .world import SimWorld

PARKED = (0.0, 0.0, -1.0)


class SimLaser(Laser):
    def __init__(self, world: SimWorld, mount_offset):
        self.world = world
        self.beam = Beam(world.model, mount_offset)
        body = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, "laser_dot")
        self._mocap = world.model.body_mocapid[body]
        self._on = False
        self.last_dot = None
        world.after_advance.append(self.update)

    def on(self) -> None:
        self._on = True
        self.update()

    def off(self) -> None:
        self._on = False
        self.update()

    @property
    def is_on(self) -> bool:
        return self._on

    def update(self) -> None:
        d = self.world.data
        pos = PARKED
        self.last_dot = None
        if self._on:
            dot = self.beam.dot(d)
            self.last_dot = dot
            if dot.point is not None:
                pos = dot.point
        d.mocap_pos[self._mocap] = pos
