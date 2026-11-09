"""Aim at many random spots in the play zone and report how close the dot lands.

    python -m lekitten.tools.aim_test --room living_room     # saves outputs/aim_test_<room>.png
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath

from lekitten.aim import Aimer
from lekitten.config import REPO, load_config
from lekitten.model import room_cfg


def random_points(poly, n, rng):
    poly = np.array(poly)
    lo, hi = poly.min(0), poly.max(0)
    path, out = MplPath(poly), []
    while len(out) < n:
        p = rng.uniform(lo, hi)
        if path.contains_point(p):
            out.append(p)
    return np.array(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--room", default=None)
    p.add_argument("--spot", default=None)
    p.add_argument("-n", type=int, default=100)
    args = p.parse_args()
    cfg = load_config()
    room = args.room or cfg["room"]
    rc = room_cfg(cfg, room)

    t0 = time.time()
    aimer = Aimer(cfg, room, args.spot)
    print(f"Aimer ready in {time.time() - t0:.1f} s ({len(aimer.lib_q):,} known-good poses)")

    rng = np.random.default_rng(1)
    targets = random_points(rc["play_zone"], args.n, rng)
    if aimer.zones.couch_play:
        for ps in rc.get("play_surfaces", []):
            targets = np.vstack([targets, random_points(ps["polygon"], args.n // 4, rng)])
    results, times, current = [], [], None
    for x, y in targets:
        t = time.time()
        r = aimer.aim_at(x, y, current)
        times.append(time.time() - t)
        results.append(r)
        if r:
            current = r.joints  # aim from wherever the arm ended up, like during play

    ok = [r for r in results if r]
    miss = np.array([r.miss_m for r in ok]) * 100
    print(f"{len(ok)}/{len(targets)} spots reached")
    if len(ok):
        print(f"miss: average {miss.mean():.2f} cm, worst {miss.max():.2f} cm")
        print(f"beam angle: {min(r.down_angle_deg for r in ok):.0f}..{max(r.down_angle_deg for r in ok):.0f} deg below horizontal")
    print(f"time per aim: average {np.mean(times) * 1000:.0f} ms, worst {np.max(times) * 1000:.0f} ms")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(mpatches.Polygon(rc["play_zone"], closed=True, fill=False, ec="#1f4e9c", lw=2))
    for ps in (rc.get("play_surfaces", []) if aimer.zones.couch_play else []):
        ax.add_patch(mpatches.Polygon(ps["polygon"], closed=True, fill=False, ec="#1f4e9c", lw=1.5, ls="--"))
    for (x, y), r in zip(targets, results):
        if r:
            ax.plot([x, r.dot[0]], [y, r.dot[1]], color="#3a9d5d", lw=1)
            ax.plot(x, y, "o", ms=4, color="#3a9d5d")
        else:
            ax.plot(x, y, "x", ms=6, color="#c0392b")
    s = rc["arm_spots"][args.spot or rc["arm_spot"]]
    ax.plot(*s["xyz"][:2], "s", color="black", ms=9)
    title = f"Aim test · {room.replace('_', ' ')} · spot {args.spot or rc['arm_spot']}\n"
    title += (f"{len(ok)}/{len(targets)} reached · average miss {miss.mean():.2f} cm · worst {miss.max():.2f} cm"
              if len(ok) else "nothing reached")
    ax.set_title(title, fontsize=11)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#3a9d5d", label="aimed (dot within 1 cm)"),
                       plt.Line2D([], [], marker="x", ls="", color="#c0392b", label="can't reach safely")],
              loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    out = REPO / f"outputs/aim_test_{room}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Saved {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
