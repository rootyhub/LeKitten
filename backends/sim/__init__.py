"""MuJoCo backend. Physics, rendering and the fake cat live only here and in sim/."""
from __future__ import annotations

import numpy as np

from lekitten.backend import Backend

from .arm import SimArm
from .scene import build_model, room_cfg
from .world import SimWorld


def make_sim_backend(cfg: dict, viewer: bool = True, video: str | None = None,
                     room: str | None = None, spot: str | None = None) -> Backend:
    if room:
        cfg = {**cfg, "room": room}
    model = build_model(cfg, spot=spot)
    world = SimWorld(model, viewer=viewer, video=video)
    rc = room_cfg(cfg)
    arm_xyz = np.array(rc["arm_spots"][spot or rc["arm_spot"]]["xyz"])
    world.set_video_camera(lookat=arm_xyz + [0, 0, 0.2], distance=1.0, azimuth=135, elevation=-25)
    arm = SimArm(world, cfg["sim"]["joint_map"])
    b = Backend(name="sim", arm=arm, wait=world.advance, close=world.close)
    b.world, b.cfg, b.room = world, cfg, rc
    return b
