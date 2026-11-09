"""Real OverheadCamera: the USB webcam through OpenCV, delivered as 640x480 RGB like the sim."""
from __future__ import annotations

import numpy as np

from lekitten.interfaces import FRAME_HEIGHT, FRAME_WIDTH, OverheadCamera

ROTATIONS = {90: "ROTATE_90_CLOCKWISE", 180: "ROTATE_180", 270: "ROTATE_90_COUNTERCLOCKWISE"}


class WebcamCamera(OverheadCamera):
    def __init__(self, index: int | None, fps: int = 30, rotation_deg: int = 0, dry_run: bool = False):
        self.dry_run = dry_run
        self.rotation = rotation_deg % 360
        if dry_run:
            print(f"[dry-run] would open webcam {index} at {FRAME_WIDTH}x{FRAME_HEIGHT} @ {fps} fps")
            return
        import cv2
        self.cv2 = cv2
        if index is None:
            raise SystemExit("real.camera_index is empty in config.yaml. Find it with: lerobot-find-cameras opencv")
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise SystemExit(f"Couldn't open webcam {index}.")

    def read(self) -> np.ndarray:
        if self.dry_run:
            return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
        ok, bgr = self.cap.read()
        if not ok:
            raise RuntimeError("Webcam read failed")
        cv2 = self.cv2
        if self.rotation:
            bgr = cv2.rotate(bgr, getattr(cv2, ROTATIONS[self.rotation]))
        if bgr.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
            bgr = cv2.resize(bgr, (FRAME_WIDTH, FRAME_HEIGHT))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
