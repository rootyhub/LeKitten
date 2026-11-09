"""Where can the laser dot go from each arm spot? Saves outputs/reach_map_<room>_<spot>.png.

    python -m lekitten.tools.reach_map --room living_room          # every candidate spot
    python -m lekitten.tools.reach_map --room living_room --spot B

Samples many arm poses inside the joint limits, throws out unsafe ones (beam not pointing down
enough, arm bumping furniture or itself), and colours each floor cell by the best beam angle that
reaches it: reachable / reachable-but-shallow (beam under 25 deg from the floor, dot smears) /
unreachable. Play zone, no-aim zones and furniture are drawn on top.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.path import Path as MplPath

from lekitten.config import REPO, load_config
from lekitten.kinematics import ArmKinematics
from lekitten.model import room_cfg
from lekitten.zones import Zones

CELL = 0.025  # 2.5 cm floor cells
AIM_JOINTS = ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos")


def room_bounds(rc: dict, margin: float = 0.6):  # noqa: D103
    pts = [p for p in rc["play_zone"]]
    for s in rc["arm_spots"].values():
        pts.append(s["xyz"][:2])
    for z in rc.get("no_aim_zones", []):
        pts += z["polygon"]
    pts = np.array(pts, float)
    lo, hi = pts.min(0) - margin, pts.max(0) + margin
    return lo, hi


def sample_reach(kin: ArmKinematics, zones: Zones, n: int, min_down: float, seed: int = 0):
    """Returns arrays of landing points (x, y) and beam angles for safe poses, plus counts.
    A landing counts if the dot is on the floor or on an enabled play surface (couch seat),
    and the beam never passes through a no-aim volume."""
    rng = np.random.default_rng(seed)
    lim = kin.jm.limits_deg()
    lo = np.array([lim[j][0] for j in AIM_JOINTS])
    hi = np.array([lim[j][1] for j in AIM_JOINTS])
    pts, angles = [], []
    stats = {"sampled": n, "collision": 0, "too_flat": 0, "no_hit": 0, "furniture": 0,
             "no_aim": 0, "floor": 0}
    fixed = {"wrist_roll.pos": 0.0, "gripper.pos": 0.0}
    for q in rng.uniform(lo, hi, size=(n, len(AIM_JOINTS))):
        kin.set_joints({**dict(zip(AIM_JOINTS, q)), **fixed})
        if kin.in_collision():
            stats["collision"] += 1
            continue
        dot = kin.dot()
        if dot.down_angle_deg < min_down:
            stats["too_flat"] += 1
            continue
        if dot.point is None:
            stats["no_hit"] += 1
            continue
        if not (dot.on_floor or zones.surface_at(dot.point)):
            stats["furniture"] += 1
            continue
        if zones.beam_crosses_no_aim(dot.origin, dot.point):
            stats["no_aim"] += 1
            continue
        stats["floor"] += 1
        pts.append(dot.point[:2])
        angles.append(dot.down_angle_deg)
    return np.array(pts).reshape(-1, 2), np.array(angles), stats


def grid_best_angle(pts, angles, lo, hi):
    w, h = np.ceil((hi - lo) / CELL).astype(int)
    best = np.full((h, w), np.nan)
    ij = np.floor((pts - lo) / CELL).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    for (i, j), a in zip(ij[ok], angles[ok]):
        if np.isnan(best[j, i]) or a > best[j, i]:
            best[j, i] = a
    return best


def fill_gaps(best):
    """Random sampling leaves holes: a cell counts as reached if a safe hit landed within one cell
    (2.5 cm) of it, taking the best beam angle among those neighbours."""
    a = np.where(np.isnan(best), -np.inf, best)
    p = np.pad(a, 1, constant_values=-np.inf)
    h, w = best.shape
    out = np.max([p[dj:dj + h, di:di + w] for dj in range(3) for di in range(3)], axis=0)
    return np.where(np.isinf(out), np.nan, out)


def zone_mask(poly, lo, shape):
    h, w = shape
    xs = lo[0] + (np.arange(w) + 0.5) * CELL
    ys = lo[1] + (np.arange(h) + 0.5) * CELL
    X, Y = np.meshgrid(xs, ys)
    return MplPath(poly).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(h, w)


def coverage(best, mask, shallow):
    n = mask.sum()
    good = np.count_nonzero(mask & (best >= shallow))
    sh = np.count_nonzero(mask & (best < shallow))  # NaN compares False
    return good / n, sh / n, 1 - (good + sh) / n


def draw(rc, spot, best, lo, hi, shallow, cov, stats, out: Path, title: str):
    cls = np.full(best.shape, 0.0)                      # 0 unreachable
    cls[best < shallow] = 1                             # 1 shallow
    cls[best >= shallow] = 2                            # 2 reachable
    cmap = ListedColormap(["#d9d9d9", "#f2b134", "#3a9d5d"])
    fig, ax = plt.subplots(figsize=(8, 8 * (hi[1] - lo[1]) / (hi[0] - lo[0]) + 1))
    ax.imshow(cls, origin="lower", extent=[lo[0], hi[0], lo[1], hi[1]], cmap=cmap, vmin=0, vmax=2,
              interpolation="nearest")
    for ob in rc.get("obstacles", []):
        cx, cy = ob["center"][:2]
        if ob["type"] == "box":
            sx, sy = ob["size"][:2]
            ax.add_patch(mpatches.Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fill=False,
                                            ec="#555", lw=1, ls="--"))
        else:
            ax.add_patch(mpatches.Circle((cx, cy), ob["radius"], fill=False, ec="#555", lw=1, ls="--"))
        ax.text(cx, cy, ob["name"].replace("_", " "), ha="center", va="center", fontsize=7, color="#333")
    for z in rc.get("no_aim_zones", []):
        ax.add_patch(mpatches.Polygon(z["polygon"], closed=True, fc="none", ec="#c0392b",
                                      hatch="///", lw=1.2))
    if rc.get("couch_play"):
        for ps in rc.get("play_surfaces", []):
            ax.add_patch(mpatches.Polygon(ps["polygon"], closed=True, fill=False, ec="#1f4e9c",
                                          lw=1.8, ls="--"))
    ax.add_patch(mpatches.Polygon(rc["play_zone"], closed=True, fill=False, ec="#1f4e9c", lw=2.5))
    s = rc["arm_spots"][spot]
    ax.plot(*s["xyz"][:2], marker="s", ms=9, color="black")
    yaw = np.radians(s["yaw_deg"])
    ax.annotate("", xy=(s["xyz"][0] + 0.25 * np.cos(yaw), s["xyz"][1] + 0.25 * np.sin(yaw)),
                xytext=s["xyz"][:2], arrowprops=dict(arrowstyle="->", lw=1.5))
    good, sh, un = cov[0]
    line2 = f"Rug: {good:.0%} reachable, {sh:.0%} shallow, {un:.0%} unreachable"
    if len(cov) > 1:
        g2, s2, u2 = cov[1]
        line2 += f"   ·   Couch seat: {g2:.0%} / {s2:.0%} / {u2:.0%}"
    ax.set_title(f"{title}\n{line2}", fontsize=11)
    legend = [mpatches.Patch(color="#3a9d5d", label=f"reachable (beam ≥ {shallow:g}° down)"),
              mpatches.Patch(color="#f2b134", label="reachable but shallow (dot smears)"),
              mpatches.Patch(color="#d9d9d9", label="unreachable"),
              mpatches.Patch(fc="none", ec="#1f4e9c", lw=2, label="play zone"),
              mpatches.Patch(fc="none", ec="#1f4e9c", ls="--", lw=1.5, label="couch seat (couch_play)"),
              mpatches.Patch(fc="none", ec="#c0392b", hatch="///", label="no-aim zone (people sitting)"),
              plt.Line2D([], [], marker="s", ls="", color="black", label="arm (arrow = facing)")]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.text(1.01, 0.0, f"{stats['sampled']:,} poses sampled\n{stats['floor']:,} land safely on the floor\n"
            f"{stats['collision']:,} bump furniture/self\n{stats['too_flat']:,} beam too flat\n"
            f"{stats['no_aim']:,} beam crosses a no-aim zone",
            transform=ax.transAxes, fontsize=7, va="bottom", color="#444")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--room", default=None)
    p.add_argument("--spot", default=None, help="One spot; default: all candidate spots")
    p.add_argument("--samples", type=int, default=500_000)
    args = p.parse_args()

    cfg = load_config()
    room = args.room or cfg["room"]
    rc = room_cfg(cfg, room)
    las = cfg["shared"]["laser"]
    lo, hi = room_bounds(rc)
    results = {}
    for spot in ([args.spot] if args.spot else list(rc["arm_spots"])):
        kin = ArmKinematics(cfg, room, spot)
        zones = Zones(rc)
        pts, ang, stats = sample_reach(kin, zones, args.samples, las["min_down_angle_deg"])
        best = fill_gaps(grid_best_angle(pts, ang, lo, hi))
        cov = [coverage(best, zone_mask(rc["play_zone"], lo, best.shape), las["shallow_angle_deg"])]
        if zones.couch_play:
            for ps in rc.get("play_surfaces", []):
                cov.append(coverage(best, zone_mask(ps["polygon"], lo, best.shape), las["shallow_angle_deg"]))
        out = REPO / f"outputs/reach_map_{room}_{spot}.png"
        surface = rc["arm_spots"][spot].get("surface", "")
        draw(rc, spot, best, lo, hi, las["shallow_angle_deg"], cov, stats, out,
             f"{room.replace('_', ' ')} · spot {spot}: {surface}")
        results[spot] = cov[0]
        extra = f"; couch seat {cov[1][0]:.0%} / {cov[1][1]:.0%} / {cov[1][2]:.0%}" if len(cov) > 1 else ""
        print(f"spot {spot} ({surface}): rug {cov[0][0]:.0%} reachable, {cov[0][1]:.0%} shallow, "
              f"{cov[0][2]:.0%} unreachable{extra}  -> {out.relative_to(REPO)}")
    if len(results) > 1:
        best_spot = max(results, key=lambda s: (results[s][0], results[s][1]))
        print(f"Best coverage: spot {best_spot}")


if __name__ == "__main__":
    main()
