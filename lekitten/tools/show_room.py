"""Open a room with the arm on a spot, or save pictures of it.

    mjpython -m lekitten.tools.show_room --room living_room            # 3D window (macOS)
    python   -m lekitten.tools.show_room --room living_room --snapshot  # outputs/room_<room>_<spot>.jpg
"""
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

from backends.sim.scene import build_model, room_cfg
from lekitten.config import REPO, load_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--room", default=None)
    p.add_argument("--spot", default=None)
    p.add_argument("--snapshot", action="store_true", help="Save pictures instead of opening a window")
    args = p.parse_args()

    cfg = load_config()
    if args.room:
        cfg["room"] = args.room
    rc = room_cfg(cfg)
    spot = args.spot or rc["arm_spot"]
    m = build_model(cfg, spot=spot)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    if not args.snapshot:
        with mujoco.viewer.launch_passive(m, d) as v:
            while v.is_running():
                mujoco.mj_step(m, d)
                v.sync()
                time.sleep(m.opt.timestep)
        return

    arm = np.array(rc["arm_spots"][spot]["xyz"])
    zone = np.array(rc["play_zone"]).mean(0)
    r = mujoco.Renderer(m, 720, 960)
    views = []
    for la, dist, az, el in [(arm + [0, 0, 0.15], 1.1, 200, -25),
                             ([zone[0], zone[1], 0.3], 3.2, 225, -45)]:
        c = mujoco.MjvCamera()
        c.lookat[:], c.distance, c.azimuth, c.elevation = la, dist, az, el
        r.update_scene(d, c)
        views.append(r.render())
    r.update_scene(d, "overhead")
    views.append(np.asarray(Image.fromarray(r.render()).resize((960, 720))))
    out = REPO / f"outputs/room_{cfg['room']}_{spot}.jpg"
    out.parent.mkdir(exist_ok=True)
    Image.fromarray(np.hstack(views)).save(out, quality=88)
    print(f"Saved {out.relative_to(REPO)}  (arm close-up | room | overhead camera)")


if __name__ == "__main__":
    main()
