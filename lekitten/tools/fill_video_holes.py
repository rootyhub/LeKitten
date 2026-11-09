"""Fill the gaps of a room scan in a sim video, for viewing only.

A phone LiDAR scan has holes where the phone never saw a surface (ceiling, high walls, behind
furniture). In the sim you see through them to the empty background. Render the video with
the background painted a key colour, then fill those pixels from the surrounding scan:

    LEKITTEN_VIDEO_BG=ff00ff LEKITTEN_VIDEO_FLOOR=3f322d LEKITTEN_VIDEO_FLOOR_BOX=-0.5,-1.4,3.1,2.75 \\
        python -m lekitten.evaluate ... --room living_room --video outputs/clip
    python -m lekitten.tools.fill_video_holes outputs/clip/teacher.mp4 outputs/clip/teacher_filled.mp4

LEKITTEN_VIDEO_FLOOR(_BOX) adds a floor slab under the room so floor gaps show floor. The camera
doesn't move, so one fill ("plate") is built from the union of all gaps and reused for every frame;
it's applied only where a frame actually shows a gap, so the arm, cat and beam stay on top.
Only for videos: training and evaluation numbers never use any of this.
"""
from __future__ import annotations

import argparse
import subprocess

import cv2
import numpy as np


def key_mask(bgr: np.ndarray) -> np.ndarray:
    """Magenta-ish pixels (strong red and blue, weak green), including the pink fringes video
    compression smears around the key colour."""
    b, g, r = (bgr[..., i].astype(int) for i in range(3))
    return ((r > 140) & (b > 140) & (g < np.minimum(r, b) - 70)).astype(np.uint8) * 255


def read_frames(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames, fps


def build_plate(frames, union: np.ndarray, scale: int = 4, blur: int = 41) -> np.ndarray:
    h, w = frames[0].shape[:2]
    # The room without moving things: per-pixel median over the clip removes the cat, beam and most arm motion.
    base = np.median(np.stack(frames[:: max(1, len(frames) // 60)]), axis=0).astype(np.uint8)
    # Keep the yellow arm out of what the fill learns from (it sits right next to a gap).
    bb, gg, rr = (base[..., i].astype(int) for i in range(3))
    arm = ((rr > 150) & (gg > 120) & (bb < 90) & (rr - bb > 90)).astype(np.uint8) * 255
    grown = cv2.dilate(np.maximum(union, arm), np.ones((15, 15), np.uint8))
    base[grown > 0] = (128, 128, 128)          # neutral, so no key colour mixes in when shrinking
    small = cv2.resize(base, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    sm = (cv2.resize(grown, (w // scale, h // scale), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8) * 255
    fill = cv2.inpaint(small, sm, 15, cv2.INPAINT_NS)
    fill = cv2.resize(fill, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(fill, (blur, blur), 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    args = p.parse_args()
    frames, fps = read_frames(args.src)
    kernel = np.ones((9, 9), np.uint8)
    masks = [cv2.dilate(key_mask(f), kernel) for f in frames]
    union = np.max(np.stack(masks), axis=0)
    plate = build_plate(frames, union)
    h, w = frames[0].shape[:2]
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p",
                            "-vcodec", "libx264", "-crf", "20", args.dst], stdin=subprocess.PIPE)
    for f, m in zip(frames, masks):
        a = cv2.GaussianBlur(m.astype(np.float32) / 255, (7, 7), 0)[..., None]
        enc.stdin.write((plate * a + f * (1 - a)).astype(np.uint8).tobytes())
    enc.stdin.close()
    enc.wait()
    print(f"Filled {len(frames)} frames ({union.mean() / 255:.0%} of the picture is scan gap at some point) -> {args.dst}")


if __name__ == "__main__":
    main()
