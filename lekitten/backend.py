"""Pick sim or real with --backend. Everything else only sees the interfaces."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable

from .config import load_config
from .interfaces import Arm, CatSource, Laser, OverheadCamera


@dataclass
class Backend:
    name: str
    arm: Arm
    camera: OverheadCamera | None = None
    laser: Laser | None = None
    cats: CatSource | None = None
    # Let time pass: sim steps physics, real sleeps.
    wait: Callable[[float], None] = field(default=lambda s: None)
    close: Callable[[], None] = field(default=lambda: None)


def add_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=["sim", "real"], default=None,
                   help="sim (MuJoCo) or real (the SO-101). Default from config.yaml.")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="Real backend only: no hardware, print what would be sent.")
    p.add_argument("--headless", action="store_true",
                   help="Sim only: no 3D window (use with --video).")
    p.add_argument("--video", default=None,
                   help="Sim only: save a 3D-view video to this .mp4 path.")
    p.add_argument("--room", default=None, help="Sim only: room from config.yaml (default: room:)")
    p.add_argument("--spot", default=None, help="Sim only: arm spot (A/B/C). Default: the room's arm_spot")


def make_backend(args: argparse.Namespace) -> Backend:
    cfg = load_config(args.config)
    name = args.backend or cfg.get("backend", "sim")
    if name == "sim":
        from backends.sim import make_sim_backend
        return make_sim_backend(cfg, viewer=not args.headless, video=args.video,
                                room=args.room, spot=args.spot, cat_seed=getattr(args, "cat_seed", 3))
    from backends.real import make_real_backend
    return make_real_backend(cfg, dry_run=args.dry_run)
