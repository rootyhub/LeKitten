"""MuJoCo backend. Physics, rendering and the fake cat live only here and in sim/."""
from __future__ import annotations

import numpy as np

from lekitten.backend import Backend
from lekitten.model import build_model, room_cfg

from .arm import SimArm
from .camera import SimCamera
from .cat import FakeCat, SimCatSource
from .laser import SimLaser
from .world import SimWorld


def make_sim_backend(cfg: dict, viewer: bool = True, video: str | None = None,
                     room: str | None = None, spot: str | None = None) -> Backend:
    if room:
        cfg = {**cfg, "room": room}
    model = build_model(cfg, spot=spot)
    world = SimWorld(model, viewer=viewer, video=video)
    rc = room_cfg(cfg)
    zone = np.array(rc["play_zone"], float)
    arm_xy = np.array(rc["arm_spots"][spot or rc["arm_spot"]]["xyz"][:2])
    mid = (zone.mean(0) + arm_xy) / 2   # frame the arm and the rug together
    world.set_video_camera(lookat=[*mid, 0.2], distance=3.4, azimuth=200, elevation=-58)
    cam = cfg["shared"]["camera"]
    laser = SimLaser(world, cfg["shared"]["laser"]["mount_offset_m"])
    cat = FakeCat(world, rc["play_zone"], rc.get("hide_spots", []), laser=laser)
    b = Backend(
        name="sim",
        arm=SimArm(world, cfg["sim"]["joint_map"]),
        camera=SimCamera(world, width=cam["width"], height=cam["height"]),
        laser=laser,
        cats=SimCatSource(cat),
        wait=world.advance,
        close=world.close,
    )
    b.world, b.cfg, b.room, b.cat = world, cfg, rc, cat
    return b
