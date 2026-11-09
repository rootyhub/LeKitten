"""Real backend: LeRobot SO-101 follower, LCUS-1 USB relay laser, OpenCV webcam.

Hardware libraries are only imported here. --dry-run needs no hardware and prints what it would send.
"""
from __future__ import annotations

import time

from lekitten.backend import Backend
from lekitten.interfaces import Cat, CatSource

from .arm import RealArm
from .camera import WebcamCamera
from .laser import RelayLaser


class NoCatsYet(CatSource):
    """Placeholder until the cat detector is trained on real footage: reports no cats,
    so the laser guard keeps the laser OFF on the real arm."""

    def get_cats(self, frame) -> list[Cat]:
        return []


def make_real_backend(cfg: dict, dry_run: bool = False) -> Backend:
    r, cam = cfg["real"], cfg["shared"]["camera"]
    arm = RealArm(cfg, dry_run=dry_run)
    laser = RelayLaser(r.get("relay_port"), dry_run=dry_run, arm_port=r.get("arm_port"))
    camera = WebcamCamera(r.get("camera_index"), fps=cam["fps"],
                          rotation_deg=r.get("camera_rotation_deg", 0), dry_run=dry_run)

    def close():
        laser.off()
        if hasattr(laser, "close"):
            laser.close()

    b = Backend(name="real", arm=arm, camera=camera, laser=laser, cats=NoCatsYet(),
                wait=time.sleep, close=close)
    b.cfg = cfg
    return b
