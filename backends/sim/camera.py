"""Sim OverheadCamera: renders the MuJoCo camera placed at the configured pose."""
from __future__ import annotations

import mujoco
import numpy as np

from lekitten.interfaces import FRAME_HEIGHT, FRAME_WIDTH, OverheadCamera

from .world import SimWorld


class SimCamera(OverheadCamera):
    def __init__(self, world: SimWorld, name: str = "overhead",
                 width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT):
        self.world = world
        self.name = name
        self._renderer = mujoco.Renderer(world.model, height=height, width=width)

    def read(self) -> np.ndarray:
        self._renderer.update_scene(self.world.data, camera=self.name)
        return self._renderer.render().copy()  # RGB uint8 (480, 640, 3)
