"""Make a smaller-image copy of a LeRobot dataset (e.g. 640x480 -> 320x240) for faster training.

    python -m lekitten.tools.downscale_dataset datasets/lekitten_sim datasets/lekitten_sim_320 --width 320 --height 240

Joints, actions, timing and episodes are copied unchanged; only the camera images are resized
(area averaging, which keeps small things like the laser dot from flickering out).
Works for sim and real recordings alike.
"""
from __future__ import annotations

import argparse
import shutil
import time

import numpy as np
from PIL import Image

from lekitten.config import REPO


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src = LeRobotDataset("local/" + args.src.rstrip("/").split("/")[-1], root=REPO / args.src)
    dst_root = REPO / args.dst
    if dst_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.dst} already exists. Add --overwrite to replace it.")
        shutil.rmtree(dst_root)

    image_keys = [k for k, f in src.features.items() if f["dtype"] in ("video", "image")]
    keep = {k: f for k, f in src.features.items()
            if k not in ("timestamp", "frame_index", "episode_index", "index", "task_index")}
    feats = {}
    for k, f in keep.items():
        f = dict(f)
        if k in image_keys:
            f["shape"] = (args.height, args.width, 3)
        feats[k] = f
    dst = LeRobotDataset.create(repo_id="local/" + args.dst.rstrip("/").split("/")[-1], fps=src.fps,
                                root=dst_root, robot_type=src.meta.robot_type, features=feats,
                                use_videos=True, image_writer_threads=4, streaming_encoding=True)

    hf = src.hf_dataset.with_format("numpy")
    ep_index = np.asarray(hf["episode_index"])
    t0 = time.time()
    for ep in range(src.num_episodes):
        idx = np.flatnonzero(ep_index == ep)
        for i in idx:
            item = src[int(i)]
            frame = {"task": item["task"]}
            for k in keep:
                v = item[k]
                if k in image_keys:  # CHW float [0,1] -> HWC uint8, resized
                    img = (v.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                    v = np.asarray(Image.fromarray(img).resize((args.width, args.height), Image.BOX))
                else:
                    v = v.numpy().astype(np.float32)
                frame[k] = v
            dst.add_frame(frame)
        dst.save_episode()
        print(f"episode {ep + 1}/{src.num_episodes}  ({len(idx)} frames, {time.time() - t0:.0f} s so far)")
    if hasattr(dst, "finalize"):
        dst.finalize()
    print(f"Saved {dst.num_frames:,} frames at {args.width}x{args.height} to {args.dst}")


if __name__ == "__main__":
    main()
