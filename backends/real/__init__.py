"""Real backend: LeRobot SO-101 follower, LCUS-1 USB relay laser, OpenCV webcam. Built in S7."""
from __future__ import annotations


def make_real_backend(cfg: dict, dry_run: bool = False):
    raise SystemExit("The real backend is written in step S7. Use --backend sim for now.")
