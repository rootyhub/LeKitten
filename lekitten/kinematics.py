"""The shared geometric model of the SO-101 in the room (MuJoCo model, no physics).

Used by aim, reach maps and the sim laser on both machines: where the beam starts, which way
it points, where the dot lands, and whether a pose bumps into furniture.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .joints import JointMap
from .model import build_model

FLOOR_Z = 0.03  # a hit below this height counts as "on the floor"


@dataclass
class Dot:
    point: np.ndarray | None      # where the beam first hits something (room frame), None = nothing
    on_floor: bool
    down_angle_deg: float         # beam angle below horizontal (negative = pointing up)
    origin: np.ndarray
    direction: np.ndarray


class Beam:
    """Laser geometry on any (model, data) that contains the arm."""

    def __init__(self, model: mujoco.MjModel, mount_offset=(0.0, 0.0, 0.0), hit_groups=(0, 1)):
        self.m = model
        # Which geom groups the beam can land on: 0 = floor/furniture looks, 1 = room scan,
        # 3 = simple collision boxes (only when there's no scan).
        self.groups = np.array([1 if g in hit_groups else 0 for g in range(6)], np.uint8)
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.arm_root = model.body_rootid[base]
        self.base_body = base
        skip_bodies = {b for b in range(model.nbody) if model.body_rootid[b] == self.arm_root}
        for name in ("laser_dot", "laser_beam"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                skip_bodies.add(bid)
        self.skip_geom = np.array([model.geom_bodyid[g] in skip_bodies for g in range(model.ngeom)])
        self.arm_geom = np.array([model.body_rootid[model.geom_bodyid[g]] == self.arm_root
                                  and model.geom_bodyid[g] != 0 for g in range(model.ngeom)])
        self.mount = np.asarray(mount_offset, float)
        self._gid = np.zeros(1, np.int32)

    def ray(self, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Beam start and unit direction: from the gripper tool point along the gripper's forward (x) axis."""
        R = d.site_xmat[self.site].reshape(3, 3)
        return d.site_xpos[self.site] + R @ self.mount, R[:, 0].copy()

    def cast(self, d: mujoco.MjData, origin, direction, max_hops: int = 20):
        """First thing the beam hits that isn't the arm itself (or the dot marker)."""
        p = np.array(origin, float)
        travelled = 0.0
        for _ in range(max_hops):
            dist = mujoco.mj_ray(self.m, d, p, direction, self.groups, 1, -1, self._gid)
            if dist < 0:
                return None, -1
            g = int(self._gid[0])
            hit = p + dist * direction
            if not self.skip_geom[g]:
                return hit, g
            p = hit + 1e-4 * direction
            travelled += dist
        return None, -1

    def dot(self, d: mujoco.MjData) -> Dot:
        o, v = self.ray(d)
        hit, _ = self.cast(d, o, v)
        down = math.degrees(math.asin(max(-1.0, min(1.0, -v[2]))))
        return Dot(hit, hit is not None and hit[2] < FLOOR_Z, down, o, v)

    def in_collision(self, d: mujoco.MjData) -> bool:
        """Any arm link (other than the base sitting on its surface) touching furniture or itself."""
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > -1e-4:
                continue
            g1, g2 = c.geom1, c.geom2
            if not (self.arm_geom[g1] or self.arm_geom[g2]):
                continue
            b1, b2 = self.m.geom_bodyid[g1], self.m.geom_bodyid[g2]
            if self.base_body in (b1, b2):
                continue
            return True
        return False


class ArmKinematics:
    """Pose the arm (LeRobot degrees) in a physics-free copy of the room and ask geometric questions."""

    def __init__(self, cfg: dict, room: str | None = None, spot: str | None = None,
                 use_scene: bool = True):
        """use_scene: land the beam on the room's real surfaces (scan / furniture models) when the
        scene loads; otherwise (e.g. home Mac without the scan) on the simple collision boxes."""
        self.model, groups = None, (0, 3)
        if use_scene:
            try:
                self.model, groups = build_model(cfg, room, spot, visual=True), (0, 1)
            except Exception as e:  # scan files missing on this machine
                print(f"(room scene not available, using simple shapes: {e})")
        if self.model is None:
            self.model = build_model(cfg, room, spot, visual=False)
        self.data = mujoco.MjData(self.model)
        self.jm = JointMap(self.model, cfg["sim"]["joint_map"])
        self.beam = Beam(self.model, cfg["shared"]["laser"]["mount_offset_m"], hit_groups=groups)

    def set_joints(self, joints: dict[str, float]) -> None:
        for j, v in joints.items():
            self.data.qpos[self.jm.qpos[j]] = self.jm.to_rad(j, v)
        mujoco.mj_fwdPosition(self.model, self.data)

    def dot(self) -> Dot:
        return self.beam.dot(self.data)

    def in_collision(self) -> bool:
        return self.beam.in_collision(self.data)
