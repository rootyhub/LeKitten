"""MuJoCo backend. Physics, rendering and the fake cat live only here and in sim/."""
from __future__ import annotations

from lekitten.backend import Backend
from lekitten.config import repo_path

from .arm import SimArm
from .world import SimWorld


def make_sim_backend(cfg: dict, viewer: bool = True, video: str | None = None) -> Backend:
    world = SimWorld(repo_path(cfg["sim"]["scene"]), viewer=viewer, video=video)
    arm = SimArm(world, cfg["sim"]["joint_map"])
    b = Backend(name="sim", arm=arm, wait=world.advance, close=world.close)
    b.world = world
    return b
