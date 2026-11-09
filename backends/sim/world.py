"""The MuJoCo world: model, physics stepping, optional 3D window and video."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


class SimWorld:
    def __init__(self, model: mujoco.MjModel, viewer: bool = True, video: str | None = None,
                 video_fps: int = 30, video_size: tuple[int, int] = (960, 720)):
        self.model = model
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.viewer = None
        if viewer:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except RuntimeError as e:
                if sys.platform == "darwin" and "mjpython" in str(e):
                    raise SystemExit(
                        "On macOS the 3D window needs mjpython. Run the same command with "
                        "'mjpython' instead of 'python', or add --headless --video outputs/x.mp4."
                    ) from e
                raise
        self._video = None
        if video:
            self._open_video(video, video_fps, video_size)
        self._next_frame_t = 0.0
        self._wall0 = None
        self.after_advance = []  # callables run after physics moves (e.g. the laser dot)

    # --- time -------------------------------------------------------------
    def advance(self, seconds: float, step_hook=None) -> None:
        """Step physics for `seconds` of sim time. Real-time paced when a window is open."""
        n = max(1, int(round(seconds / self.model.opt.timestep)))
        for _ in range(n):
            if step_hook:
                step_hook()
            mujoco.mj_step(self.model, self.data)
            if self._video and self.data.time >= self._next_frame_t:
                self._run_hooks()
                self._write_frame()
                self._next_frame_t += 1.0 / self._video_fps
        self._run_hooks()
        if self.viewer:
            if not self.viewer.is_running():
                raise SystemExit("3D window closed.")
            self.viewer.sync()
            if self._wall0 is None:
                self._wall0 = time.monotonic() - self.data.time
            lag = self.data.time - (time.monotonic() - self._wall0)
            if lag > 0:
                time.sleep(lag)

    def _run_hooks(self) -> None:
        mujoco.mj_kinematics(self.model, self.data)
        for f in self.after_advance:
            f()

    # --- video ------------------------------------------------------------
    def _open_video(self, path: str, fps: int, size: tuple[int, int]) -> None:
        w, h = size
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, w)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, h)
        self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        self._video_opt = mujoco.MjvOption()
        self._video_opt.geomgroup[5] = 1  # show display-only helpers (the beam line) in 3D videos
        self._cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self._cam)
        self._cam.distance, self._cam.azimuth, self._cam.elevation = 0.9, 135, -25
        self._cam.lookat[:] = [0.0, 0.0, 0.2]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._video_fps = fps
        self._video = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p",
             "-vcodec", "libx264", path],
            stdin=subprocess.PIPE,
        )
        self.video_path = path

    def set_video_camera(self, lookat, distance, azimuth, elevation) -> None:
        if self._video:
            self._cam.lookat[:] = lookat
            self._cam.distance, self._cam.azimuth, self._cam.elevation = distance, azimuth, elevation

    def _write_frame(self) -> None:
        self._renderer.update_scene(self.data, self._cam, scene_option=self._video_opt)
        self._video.stdin.write(np.ascontiguousarray(self._renderer.render()).tobytes())

    def close(self) -> None:
        if self._video:
            self._video.stdin.close()
            self._video.wait()
            self._video = None
            print(f"Saved video: {self.video_path}")
        if self.viewer:
            self.viewer.close()
            self.viewer = None
