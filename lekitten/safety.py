"""Safety rules. Every command goes through here, in sim and on the real arm.

Rules (from CLAUDE.md):
  1. The beam always points below horizontal by at least min_down_angle_deg.
  2. The dot is never inside a cat's radius + cat_margin_m, and the beam never passes through
     or lands in a no-aim zone (couch seating, where people's heads are).
  3. The laser defaults to OFF, and turns off when no cat is seen for no_cat_timeout_s, on any
     error, on exit, and after max_session_s.
  4. Joint targets stay inside the joint limits and joints move no faster than max_joint_speed_deg_s.
  5. Sessions end by landing the dot on the toy spot, holding, then laser off (play.py does the
     landing; the guard enforces the hard stop).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .interfaces import JOINTS, Cat, Laser
from .kinematics import ArmKinematics
from .zones import Zones


@dataclass
class Verdict:
    ok: bool
    reason: str = ""


class Safety:
    def __init__(self, cfg: dict, zones: Zones, kin: ArmKinematics):
        s, las, arm = cfg["shared"]["safety"], cfg["shared"]["laser"], cfg["shared"]["arm"]
        self.zones, self.kin = zones, kin
        self.cat_margin = s["cat_margin_m"]
        self.min_down = las["min_down_angle_deg"]
        self.max_speed = arm["max_joint_speed_deg_s"]
        lim = kin.jm.limits_deg()
        user = arm.get("joint_limits_deg") or {}
        self.limits = {j: tuple(user.get(j, lim[j])) for j in JOINTS}

    # --- rule 2: where the dot may go ---------------------------------------
    def target_ok(self, x: float, y: float, cats: list[Cat]) -> Verdict:
        for c in cats:
            if math.hypot(x - c.x, y - c.y) < c.radius + self.cat_margin:
                return Verdict(False, "too close to a cat")
        return Verdict(True)

    # --- rules 1, 2, 4: the full check of a pose before the laser may be on ---
    def pose_ok(self, joints: dict[str, float], cats: list[Cat]) -> Verdict:
        for j, v in joints.items():
            lo, hi = self.limits[j]
            if not lo - 1e-6 <= v <= hi + 1e-6:
                return Verdict(False, f"{j} outside its limits")
        self.kin.set_joints(joints)
        if self.kin.in_collision():
            return Verdict(False, "arm would bump something")
        dot = self.kin.dot()
        if dot.down_angle_deg < self.min_down:
            return Verdict(False, f"beam only {dot.down_angle_deg:.0f} deg below horizontal")
        if dot.point is None:
            return Verdict(False, "beam hits nothing")
        if not (dot.on_floor or self.zones.surface_at(dot.point)):
            return Verdict(False, "dot not on a play surface")
        zone = self.zones.beam_crosses_no_aim(dot.origin, dot.point)
        if zone:
            return Verdict(False, f"beam crosses no-aim zone '{zone}'")
        v = self.target_ok(dot.point[0], dot.point[1], cats)
        if not v.ok:
            return v
        if self.beam_near_cat(dot.origin, dot.point, cats):
            return Verdict(False, "beam passes through a cat")
        return Verdict(True)

    def beam_near_cat(self, origin, hit, cats: list[Cat], cat_height: float = 0.45,
                      step: float = 0.02) -> bool:
        """The beam must not pass through a cylinder (cat radius + margin, up to head height) around any cat."""
        if not cats:
            return False
        origin, hit = np.asarray(origin, float), np.asarray(hit, float)
        n = max(2, int(np.linalg.norm(hit - origin) / step) + 1)
        seg = origin + np.linspace(0, 1, n)[:, None] * (hit - origin)
        low = seg[seg[:, 2] <= cat_height]
        for c in cats:
            if len(low) and (np.hypot(low[:, 0] - c.x, low[:, 1] - c.y) < c.radius + self.cat_margin).any():
                return True
        return False

    # --- rule 4: speed ---------------------------------------------------------
    def limit_speed(self, current: dict[str, float], target: dict[str, float], dt: float) -> dict[str, float]:
        step = self.max_speed * dt
        out = {}
        for j, v in target.items():
            lo, hi = self.limits[j]
            v = min(max(v, lo), hi)
            c = current.get(j, v)
            out[j] = c + max(-step, min(step, v - c))
        return out


@dataclass
class LaserGuard:
    """Rule 3: owns the laser. Off by default, off on timeout / error / exit / session end."""
    laser: Laser
    no_cat_timeout_s: float
    max_session_s: float
    clock: callable = time.monotonic
    _start: float = field(default=None, init=False)
    _last_cat: float = field(default=None, init=False)

    def __enter__(self):
        self.laser.off()
        self._start = self.clock()
        self._last_cat = None
        return self

    def __exit__(self, *exc):
        self.laser.off()  # always, including on errors and Ctrl+C
        return False

    @property
    def elapsed(self) -> float:
        return self.clock() - self._start

    @property
    def session_over(self) -> bool:
        return self.elapsed >= self.max_session_s

    def saw_cats(self, cats: list[Cat]) -> None:
        if cats:
            self._last_cat = self.clock()

    @property
    def cats_recent(self) -> bool:
        return self._last_cat is not None and self.clock() - self._last_cat <= self.no_cat_timeout_s

    def set(self, want_on: bool, verdict: Verdict | None = None) -> str:
        """Turn the laser on only if wanted AND every rule allows it. Returns why it's off ('' if on)."""
        reason = ""
        if not want_on:
            reason = "not needed"
        elif self.session_over:
            reason = "session over"
        elif not self.cats_recent:
            reason = "no cat seen recently"
        elif verdict is not None and not verdict.ok:
            reason = verdict.reason
        if reason:
            if self.laser.is_on:
                self.laser.off()
        elif not self.laser.is_on:
            self.laser.on()
        return reason


def clamp_to_zone(p: np.ndarray, zones: Zones) -> np.ndarray:
    """Pull a floor target back inside the play zone (keeps behaviours from wandering off the rug)."""
    poly = zones.play.polygon
    lo, hi = poly.min(0) + 0.05, poly.max(0) - 0.05
    return np.clip(p, lo, hi)
