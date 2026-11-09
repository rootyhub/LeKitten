"""Record the hand-written player (the teacher) in the sim as a LeRobot dataset.

    python -m lekitten.record --episodes 2 --duration 10            # quick test
    python -m lekitten.record --episodes 50 --duration 45           # the training set (T2)

Each episode: a fresh sim, a different fake-cat seed and starting pose, one play session that ends
calmly (dot holds still, laser off). Every tick (30 per second) saves both camera images, the arm's joints and the commanded
joints, with the same names and units the real arm uses with lerobot-record.

The dataset stays on this laptop in datasets/ (git-ignored): its images show the owner's home.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

from .aim import Aimer
from .config import REPO, load_config
from .interfaces import JOINTS
from .play import build_parser, run

FPS = 30
TASK = "Lead the cat around the rug with the laser dot."


def features(h: int, w: int) -> dict:
    img = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    vec = {"dtype": "float32", "shape": (len(JOINTS),), "names": list(JOINTS)}
    return {"observation.state": vec, "action": dict(vec),
            "observation.images.overhead": dict(img), "observation.images.wrist": dict(img)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--duration", type=float, default=45.0, help="Play time per episode before the calm ending")
    p.add_argument("--root", default="datasets/lekitten_sim")
    p.add_argument("--room", default="living_room")
    p.add_argument("--first-seed", type=int, default=100)
    p.add_argument("--overwrite", action="store_true", help="Delete an existing dataset at --root first")
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from backends.sim.camera import SimCamera

    cfg = load_config()
    cam = cfg["shared"]["camera"]
    root = REPO / args.root
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.root} already exists. Add --overwrite to replace it.")
        shutil.rmtree(root)
    ds = LeRobotDataset.create(repo_id="local/lekitten_sim", fps=FPS, root=root, robot_type="so101_follower",
                               features=features(cam["height"], cam["width"]), use_videos=True,
                               image_writer_threads=4, streaming_encoding=True)

    cfg["room"] = args.room
    print("Preparing aim (a few seconds)...")
    aimer = Aimer(cfg)
    rng = np.random.default_rng(args.first_seed)
    summary = []
    for ep in range(args.episodes):
        seed = args.first_seed + ep
        # A different, safe starting pose each time: a random known-good aiming pose.
        q = aimer.lib_q[rng.integers(len(aimer.lib_q))]
        start = {**dict(zip(("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos"), map(float, q))),
                 "wrist_roll.pos": 0.0, "gripper.pos": 0.0}
        pa = build_parser().parse_args(["--backend", "sim", "--headless", "--room", args.room,
                                        "--duration", str(args.duration), "--seed", str(seed)])
        pa.cat_seed = seed
        wrist = {}
        n = {"frames": 0}

        def on_tick(b, state, frame, action):
            if "cam" not in wrist:
                wrist["cam"] = SimCamera(b.world, "wrist_cam", width=cam["width"], height=cam["height"])
            ds.add_frame({
                "observation.state": np.array([state[j] for j in JOINTS], np.float32),
                "action": np.array([action[j] for j in JOINTS], np.float32),
                "observation.images.overhead": frame,
                "observation.images.wrist": wrist["cam"].read(),
                "task": TASK,
            })
            n["frames"] += 1

        t0 = time.time()
        stats = run(pa, on_tick=on_tick, rate_hz=FPS, aimer=aimer, start_pose=start, verbose=False)
        ds.save_episode()
        on = stats["laser_on"] / max(1, stats["ticks"])
        summary.append((ep, seed, n["frames"], on, stats["ended_cleanly"], time.time() - t0))
        print(f"episode {ep:3d}  seed {seed}  {n['frames']:5d} frames  laser on {on:4.0%}  "
              f"{'ended cleanly' if stats['ended_cleanly'] else 'DID NOT end cleanly'}  {time.time() - t0:5.1f} s")

    ds.finalize() if hasattr(ds, "finalize") else None
    frames = sum(s[2] for s in summary)
    print(f"\nSaved {len(summary)} episodes, {frames:,} frames ({frames / FPS / 60:.1f} min) to {args.root}")
    size = sum(f.stat().st_size for f in Path(root).rglob("*") if f.is_file())
    print(f"Dataset size on disk: {size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
