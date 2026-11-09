"""Sim Laser: shows a red dot where the beam first hits the floor or furniture, only while on."""
from __future__ import annotations

import mujoco
import numpy as np

from lekitten.interfaces import Laser
from lekitten.kinematics import Beam

from .world import SimWorld

PARKED = (0.0, 0.0, -1.0)


class SimLaser(Laser):
    def __init__(self, world: SimWorld, mount_offset):
        self.world = world
        self.beam = Beam(world.model, mount_offset)
        m = world.model
        self._mocap = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "laser_dot")]
        self._beam = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "laser_beam")]
        self._beam_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "laser_beam")
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
        pos, beam_pos = PARKED, PARKED
        self.last_dot = None
        if self._on:
            dot = self.beam.dot(d)
            self.last_dot = dot
            if dot.point is not None:
                pos = dot.point
                # Beam cylinder: centred between gripper and dot, z axis along the beam.
                v = dot.point - dot.origin
                length = float(np.linalg.norm(v))
                beam_pos = dot.origin + v / 2
                q = np.zeros(4)
                mujoco.mju_quatZ2Vec(q, v / max(length, 1e-9))
                d.mocap_quat[self._beam] = q
                self.world.model.geom_size[self._beam_geom][1] = length / 2
        d.mocap_pos[self._mocap] = pos
        d.mocap_pos[self._beam] = beam_pos
