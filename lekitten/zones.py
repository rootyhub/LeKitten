"""Room zones in plain geometry (no MuJoCo): play zone, play surfaces, no-aim volumes.

Shared by aim, safety and the reach maps on both machines.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as _Path


@dataclass
class Volume:
    name: str
    polygon: np.ndarray
    z_min: float
    z_max: float

    def contains(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        inside = _Path(self.polygon).contains_points(pts[:, :2])
        return inside & (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max)


@dataclass
class Surface:
    name: str
    polygon: np.ndarray
    z: float
    tol: float = 0.06  # how far a hit may be from the nominal height (scan bumps, cushions)

    def contains(self, p) -> bool:
        return bool(_Path(self.polygon).contains_point(p[:2])) and abs(p[2] - self.z) <= self.tol


class Zones:
    def __init__(self, room: dict):
        self.couch_play = bool(room.get("couch_play", False))
        self.play = Surface("floor", np.array(room["play_zone"], float), 0.0, tol=0.03)
        self.surfaces = [self.play]
        if self.couch_play:
            self.surfaces += [Surface(s["name"], np.array(s["polygon"], float), s["z"])
                              for s in room.get("play_surfaces", [])]
        self.no_aim = []
        for z in room.get("no_aim_zones", []):
            zmin = z.get("z_min", 0.0) if self.couch_play else z.get("z_min_no_couch_play", z.get("z_min", 0.0))
            self.no_aim.append(Volume(z["name"], np.array(z["polygon"], float), zmin, z["z_max"]))
        self.toy = np.array(room["toy_spot"], float)

    def surface_at(self, p) -> Surface | None:
        """The play surface a 3D dot position lies on, or None."""
        for s in self.surfaces:
            if s.contains(p):
                return s
        return None

    def beam_crosses_no_aim(self, origin, hit, step: float = 0.02) -> str | None:
        """Name of the first no-aim volume the beam segment origin->hit passes through (or ends in)."""
        origin, hit = np.asarray(origin, float), np.asarray(hit, float)
        n = max(2, int(np.linalg.norm(hit - origin) / step) + 1)
        seg = origin + np.linspace(0, 1, n)[:, None] * (hit - origin)
        for v in self.no_aim:
            if v.contains(seg).any():
                return v.name
        return None
