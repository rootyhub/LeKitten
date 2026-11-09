"""Play with the cats. Uses only the interfaces, so it runs unchanged in sim and on the real arm.

    python -m lekitten.play --backend sim --headless --video outputs/play_3d.mp4 \\
        --camera-video outputs/play_overhead.mp4 --duration 120
    mjpython -m lekitten.play --backend sim          # watch it live in the 3D window (macOS)
    python -m lekitten.play --backend real           # the real arm (S7)

Behaviours:
  lead_ahead      the dot stays a little ahead of the cat, so it chases
  circle_and_pause the dot circles slowly, then freezes (cats love a still dot)
  dart_away       laser off, jump somewhere else, laser back on
  land_on_toy     end of session: dot goes to the toy, holds, laser off
Every pose goes through safety.py first; the laser is only on when every rule passes.
"""
from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np

from .aim import Aimer
from .backend import add_backend_args, make_backend
from .config import load_config
from .model import room_cfg
from .safety import LaserGuard, Safety, clamp_to_zone
from .zones import Zones

RATE_HZ = 20


class Behaviours:
    """Chooses where the dot should go next. Pure geometry: no hardware, no MuJoCo."""

    def __init__(self, zones: Zones, play_cfg: dict, rng: np.random.Generator,
                 arm_xy=(0.0, 0.0), clearance: float = 0.4):
        self.zones, self.rng = zones, rng
        # The beam comes from the arm; a dot on the far side of the cat means the beam passes
        # over the cat, which safety refuses. So targets are picked with a clear line from the arm.
        self.arm_xy = np.asarray(arm_xy, float)
        self.clearance = clearance
        self.lead = play_cfg["lead_distance_m"]
        self.speed = play_cfg["dot_speed_m_s"]
        self.mode, self.until = "lead_ahead", 0.0
        self.heading = rng.uniform(0, 2 * math.pi)
        self.center = None
        self.dart_goal = None

    def clear_line(self, c: np.ndarray, p: np.ndarray) -> bool:
        """True if the cat at c is at least `clearance` from the arm->p line (seen from above)."""
        a, d = self.arm_xy, p - self.arm_xy
        t = np.clip(np.dot(c - a, d) / max(np.dot(d, d), 1e-9), 0.0, 1.0)
        return np.linalg.norm(a + t * d - c) >= self.clearance or np.linalg.norm(p - c) < 0.05

    def _next_mode(self, now: float) -> None:
        r = self.rng.random()
        if r < 0.5:
            self.mode, self.until = "lead_ahead", now + self.rng.uniform(5, 10)
        elif r < 0.8:
            self.mode, self.until = "circle_and_pause", now + self.rng.uniform(4, 7)
            self.center = None
        else:
            self.mode, self.until = "dart_away", now + 1.2
            self.dart_goal = None

    def force_dart(self, now: float) -> None:
        """Safety keeps blocking the laser (cat in the way, can't aim): jump somewhere clear."""
        if self.mode != "dart_away":
            self.mode, self.until, self.dart_goal = "dart_away", now + 1.2, None

    def set_reachable(self, pts: np.ndarray) -> None:
        """Floor points the arm is known to reach (from the aimer's pose library)."""
        self.reach = np.asarray(pts, float).reshape(-1, 2)

    def snap(self, p: np.ndarray) -> np.ndarray:
        if getattr(self, "reach", None) is None or not len(self.reach):
            return p
        d = np.linalg.norm(self.reach - p, axis=1)
        i = int(d.argmin())
        return p if d[i] < 0.05 else self.reach[i].copy()

    def goal(self, now: float, dot: np.ndarray, cat) -> tuple[np.ndarray, bool]:
        """Returns (where the dot should head, whether the laser should be on), snapped to reachable spots."""
        g, on = self._goal(now, dot, cat)
        return self.snap(g), on

    def _goal(self, now: float, dot: np.ndarray, cat) -> tuple[np.ndarray, bool]:
        if now >= self.until:
            self._next_mode(now)
        zone = self.zones.play.polygon
        if cat is None:  # no cat in view: drift slowly around the middle of the rug
            return clamp_to_zone(zone.mean(0) + 0.3 * np.array([math.cos(now / 3), math.sin(now / 3)]),
                                 self.zones), True
        c = np.array([cat.x, cat.y])
        # Caught! When the cat gets on top of the dot, the dot vanishes and reappears elsewhere.
        if self.mode != "dart_away" and np.linalg.norm(dot - c) < self.clearance:
            self.mode, self.until, self.dart_goal = "dart_away", now + 1.2, None
        if self.mode == "lead_ahead":
            # Keep the dot `lead` m from the cat; turn the heading slowly so it runs in curves,
            # and steer back inward when the dot would leave the rug.
            self.heading += self.rng.normal(0, 0.15)
            g = c + self.lead * np.array([math.cos(self.heading), math.sin(self.heading)])
            if not np.allclose(clamp_to_zone(g, self.zones), g) or not self.clear_line(c, g):
                # Turn toward the nearest heading that stays on the rug with a clear beam.
                for k in range(1, 13):
                    for sgn in (1, -1):
                        h = self.heading + sgn * k * math.pi / 12
                        cand = c + self.lead * np.array([math.cos(h), math.sin(h)])
                        if np.allclose(clamp_to_zone(cand, self.zones), cand) and self.clear_line(c, cand):
                            self.heading, g = h, cand
                            break
                    else:
                        continue
                    break
                g = clamp_to_zone(g, self.zones)
            return g, True
        if self.mode == "circle_and_pause":
            if self.center is None:
                self.center = clamp_to_zone(dot.copy(), self.zones)
            left = self.until - now
            if left < 1.8:  # freeze
                return dot, True
            a = now * 1.6
            return clamp_to_zone(self.center + 0.18 * np.array([math.cos(a), math.sin(a)]), self.zones), True
        # dart_away: laser off while it jumps to a spot far from the cat
        if self.dart_goal is None:
            for _ in range(50):
                p = self.rng.uniform(zone.min(0) + 0.1, zone.max(0) - 0.1)
                p = self.snap(p)
                if np.linalg.norm(p - c) > 0.8 and self.clear_line(c, p):
                    self.dart_goal = p
                    break
            else:
                self.dart_goal = zone.mean(0)
        on = np.linalg.norm(dot - self.dart_goal) < 0.05
        return self.dart_goal, on


def step_toward(p: np.ndarray, goal: np.ndarray, max_step: float) -> np.ndarray:
    d = goal - p
    n = np.linalg.norm(d)
    return goal.copy() if n <= max_step else p + d / n * max_step


class FrameWriter:
    def __init__(self, path: str, w: int, h: int, fps: int):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p",
                                   "-vcodec", "libx264", path], stdin=subprocess.PIPE)
        self.path = path

    def write(self, frame: np.ndarray) -> None:
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.p.stdin.close()
        self.p.wait()
        print(f"Saved video: {self.path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_backend_args(p)
    p.add_argument("--duration", type=float, default=None, help="Session length in s (default: config play.session_s)")
    p.add_argument("--camera-video", default=None, help="Save the overhead camera's view to this .mp4")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.room:
        cfg["room"] = args.room
    rc = room_cfg(cfg)
    s, play = cfg["shared"]["safety"], cfg["shared"]["play"]
    duration = min(args.duration or play["session_s"], s["max_session_s"])
    rng = np.random.default_rng(args.seed)

    print("Preparing aim (a few seconds)...")
    aimer = Aimer(cfg, spot=args.spot)
    zones = aimer.zones
    safety = Safety(cfg, zones, aimer.kin)
    spot = rc["arm_spots"][args.spot or rc["arm_spot"]]
    behaviours = Behaviours(zones, play, rng, arm_xy=spot["xyz"][:2],
                            clearance=0.2 + cfg["shared"]["safety"]["cat_margin_m"] + 0.05)
    lib = aimer.lib_p
    on_rug = (lib[:, 2] < 0.03) & np.array([zones.play.contains(q) for q in lib]) if len(lib) else []
    behaviours.set_reachable(lib[on_rug, :2] if len(lib) else lib)
    b = make_backend(args)
    sim_clock = (lambda: b.world.data.time) if b.name == "sim" else None
    cam_writer = None
    if args.camera_video:
        c = cfg["shared"]["camera"]
        cam_writer = FrameWriter(args.camera_video, c["width"], c["height"], RATE_HZ)

    dt = 1.0 / RATE_HZ
    stats = {"ticks": 0, "laser_on": 0, "blocked": {}}
    guard_kw = dict(no_cat_timeout_s=s["no_cat_timeout_s"], max_session_s=duration + s["toy_hold_s"] + 10)
    if sim_clock:
        guard_kw["clock"] = sim_clock
    try:
        b.arm.connect()
        with LaserGuard(b.laser, **guard_kw) as guard:
            current = b.arm.read_joints()
            dot = zones.play.polygon.mean(0)
            phase, hold_until = "play", None
            blocked_since = None
            while True:
                now = guard.elapsed
                frame = b.camera.read()
                if cam_writer:
                    cam_writer.write(frame)
                cats = b.cats.get_cats(frame)
                guard.saw_cats(cats)
                cat = cats[0] if cats else None

                if phase == "play" and now >= duration:
                    phase = "toy"
                if phase == "play":
                    goal, want_on = behaviours.goal(now, dot, cat)
                    # Never steer the dot into a cat: if the goal is too close, push it out.
                    for c in cats:
                        v = goal - [c.x, c.y]
                        dmin = c.radius + safety.cat_margin + 0.05
                        if np.linalg.norm(v) < dmin:
                            goal = np.array([c.x, c.y]) + (v / (np.linalg.norm(v) + 1e-9)) * dmin
                    jump = behaviours.mode == "dart_away" and not want_on
                    dot = goal.copy() if jump else step_toward(dot, goal, play["dot_speed_m_s"] * dt)
                else:  # land on the toy, hold, then finish
                    want_on = True
                    dot = step_toward(dot, zones.toy, play["dot_speed_m_s"] * dt)
                    if np.linalg.norm(dot - zones.toy) < 0.01:
                        hold_until = hold_until or now + s["toy_hold_s"]
                        if now >= hold_until:
                            guard.set(False)
                            print("Landed on the toy, held, laser off. Session done.")
                            break
                    if now > duration + 15:
                        print("Couldn't reach the toy in time; laser off.")
                        break

                aim = aimer.aim_at(dot[0], dot[1], current)
                verdict = None
                if aim is not None:
                    target = safety.limit_speed(current, aim.joints, dt)
                    verdict = safety.pose_ok(target, cats)
                    b.arm.send_joints(target)
                    current = target
                why = guard.set(want_on and aim is not None, verdict)
                if phase == "play" and want_on and why:
                    blocked_since = blocked_since if blocked_since is not None else now
                    if now - blocked_since > 0.4:
                        behaviours.force_dart(now)
                        blocked_since = None
                else:
                    blocked_since = None
                stats["ticks"] += 1
                stats["laser_on"] += b.laser.is_on
                if why and want_on:
                    key = "can't aim there" if aim is None else why
                    stats["blocked"][key] = stats["blocked"].get(key, 0) + 1
                b.wait(dt)
        b.arm.disconnect()
    finally:
        if cam_writer:
            cam_writer.close()
        b.close()
    print(f"Laser on {stats['laser_on'] / max(1, stats['ticks']):.0%} of the time.")
    if stats["blocked"]:
        print("Laser held off because:", ", ".join(f"{k} ({v} ticks)" for k, v in
                                                   sorted(stats["blocked"].items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
