"""Teacher vs student: same cats, same starting poses, same safety rules.

    python -m lekitten.evaluate --policy outputs/train/t4/checkpoints/last/pretrained_model --episodes 20
    python -m lekitten.evaluate --policy ... --episodes 1 --video outputs/compare     # + side-by-side video

Measured during the play time of each episode (the calm ending is the same for both):
  laser on        share of time every safety rule allowed the laser on
  refused share   share of play time safety held the laser off, by reason
  dot-cat         median distance from the dot to the cat while the laser is on
  in the sweet spot share of laser-on time with the dot 0.35-0.70 m from the cat (close enough to chase,
                  outside the safety margin)
  ended cleanly   the session ended correctly (dot held still, laser off)
  jitter          average size of frame-to-frame changes in joint speed (deg/frame^2); lower is smoother
Writes outputs/eval_<name>.json for the Training Lab page.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from .aim import Aimer
from .config import REPO, load_config
from .play import build_parser, run
from .policy_play import Student, reason_group, run_student

SWEET = (0.35, 0.70)


class Meter:
    """Collects per-tick numbers during the play part of an episode (via the on_tick hook)."""

    def __init__(self, duration: float):
        self.duration, self.t0 = duration, None
        self.ticks = self.on = 0
        self.dist = []
        self.cmds = []  # commanded aiming joints, for the jitter score

    def __call__(self, b, state, frame, action):
        t = b.world.data.time
        self.t0 = t if self.t0 is None else self.t0
        if t - self.t0 >= self.duration:
            return
        self.ticks += 1
        self.cmds.append([action[j] for j in ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos")])
        if b.laser.is_on:
            self.on += 1
            d = b.laser.last_dot
            if d is not None and d.point is not None and not b.cat.hidden:
                self.dist.append(math.hypot(d.point[0] - b.cat.pos[0], d.point[1] - b.cat.pos[1]))


def summarize(rows: list[dict]) -> dict:
    ticks = sum(r["ticks"] for r in rows)
    on = sum(r["on"] for r in rows)
    dist = np.concatenate([np.array(r["dist"]) for r in rows if r["dist"]] or [np.array([])])
    blocked = {}
    for r in rows:
        for k, v in r["blocked"].items():
            blocked[k] = blocked.get(k, 0) + v
    jit = [float(np.abs(np.diff(np.array(r["cmds"]), 2, axis=0)).max(1).mean()) for r in rows if len(r["cmds"]) > 3]
    return {
        "episodes": len(rows),
        "jitter": float(np.mean(jit)) if jit else None,
        "laser_on": on / max(1, ticks),
        "median_dot_cat_m": float(np.median(dist)) if len(dist) else None,
        "sweet_spot": float(((dist >= SWEET[0]) & (dist <= SWEET[1])).mean()) if len(dist) else None,
        "ended_cleanly": sum(r["clean"] for r in rows) / len(rows),
        "refused_share": {k: v / max(ticks, 1) for k, v in sorted(blocked.items(), key=lambda kv: -kv[1])},
        "per_episode_laser_on": [r["on"] / max(1, r["ticks"]) for r in rows],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--duration", type=float, default=45.0)
    p.add_argument("--first-seed", type=int, default=1000, help="Different from the training seeds (100-149)")
    p.add_argument("--name", default="eval")
    p.add_argument("--room", default=None, help="Room from config.yaml (default: room:, the public sample_room)")
    p.add_argument("--video", default=None, help="Folder: save teacher/student 3D videos of episode 0 + side by side")
    p.add_argument("--ensemble", type=float, default=None, help="Student uses temporal ensembling (e.g. 0.01; negative favours newer plans)")
    p.add_argument("--n-action-steps", type=int, default=None, help="Student re-plans every N frames (trained with 10)")
    p.add_argument("--skip-teacher", action="store_true", help="Only run the student (teacher results don't change)")
    args = p.parse_args()

    cfg = load_config()
    cfg["room"] = args.room or cfg["room"]
    print("Loading the student and preparing aim...")
    student = Student(args.policy, ensemble=args.ensemble, n_action_steps=args.n_action_steps)
    aimer = Aimer(cfg)
    rng = np.random.default_rng(args.first_seed)
    rows = {"teacher": [], "student": []}
    for ep in range(args.episodes):
        seed = args.first_seed + ep
        q = aimer.lib_q[rng.integers(len(aimer.lib_q))]
        start = {"shoulder_pan.pos": float(q[0]), "shoulder_lift.pos": float(q[1]), "elbow_flex.pos": float(q[2]),
                 "wrist_flex.pos": float(q[3]), "wrist_roll.pos": 0.0, "gripper.pos": 0.0}
        for who in (("student",) if args.skip_teacher else ("teacher", "student")):
            argv = ["--backend", "sim", "--headless", "--room", cfg["room"],
                    "--duration", str(args.duration), "--seed", str(seed)]
            if args.video and ep == 0:
                Path(REPO / args.video).mkdir(parents=True, exist_ok=True)
                argv += ["--video", str(REPO / args.video / f"{who}.mp4")]
            pa = build_parser().parse_args(argv)
            pa.cat_seed = seed
            m = Meter(args.duration)
            if who == "teacher":
                st = run(pa, on_tick=m, rate_hz=30, aimer=aimer, start_pose=start, verbose=False)
                blocked = {}
                for k, v in st["blocked"].items():
                    blocked[reason_group(k)] = blocked.get(reason_group(k), 0) + v
            else:
                st = run_student(student, pa, aimer, start_pose=start, on_tick=m, verbose=False)
                blocked = st["blocked"]
            rows[who].append({"ticks": m.ticks, "on": m.on, "dist": m.dist, "blocked": blocked,
                              "clean": st["ended_cleanly"], "cmds": m.cmds})
        s = rows["student"][-1]
        t = rows["teacher"][-1] if rows["teacher"] else None
        print(f"episode {ep:2d}  seed {seed}  laser on: "
              + (f"teacher {t['on'] / max(1, t['ticks']):4.0%}  " if t else "")
              + f"student {s['on'] / max(1, s['ticks']):4.0%}", flush=True)

    out = {who: summarize(r) for who, r in rows.items() if r}
    out["settings"] = {"ensemble": args.ensemble, "n_action_steps": args.n_action_steps}
    out["policy"] = args.policy
    path = REPO / f"outputs/eval_{args.name}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_episode_laser_on"} if isinstance(v, dict) else v
                      for k, v in out.items()}, indent=2))
    print(f"Saved {path.relative_to(REPO)}")

    if args.video and not args.skip_teacher:
        d = REPO / args.video
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(d / "teacher.mp4"), "-i", str(d / "student.mp4"),
                        "-filter_complex",
                        "[0:v]drawtext=text='Teacher (hand-written)':x=20:y=20:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5[a];"
                        "[1:v]drawtext=text='Student (ACT policy)':x=20:y=20:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5[b];"
                        "[a][b]hstack", "-pix_fmt", "yuv420p", str(d / "side_by_side.mp4")], check=False)
        print(f"Saved {args.video}/side_by_side.mp4")


if __name__ == "__main__":
    main()
