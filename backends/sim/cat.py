"""A fake cat for the sim: wanders, pauses, dashes, stalks and pounces at the dot, sometimes hides.

Its body is a mocap marker in the scene (not hit by the laser ray, like a real cat shouldn't be).
The sim CatSource returns its true position (the real one will use a detector on camera frames).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
from matplotlib.path import Path as MplPath

from lekitten.interfaces import Cat, CatSource

from .world import SimWorld

CAT_RADIUS = 0.2
SPEED = {"wander": 0.25, "dash": 1.4, "stalk": 0.15, "pounce": 2.2, "to_hide": 0.6}


class FakeCat:
    def __init__(self, world: SimWorld, play_zone, hide_spots, laser=None, seed: int = 3):
        self.world = world
        self.zone = MplPath(np.array(play_zone, float))
        self.lo, self.hi = np.array(play_zone).min(0), np.array(play_zone).max(0)
        self.hide_spots = [np.array(h, float) for h in hide_spots]
        self.laser = laser
        self.rng = np.random.default_rng(seed)
        body = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, "cat")
        self._mocap = world.model.body_mocapid[body]
        self.pos = (self.lo + self.hi) / 2 + self.rng.uniform(-0.2, 0.2, 2)
        self.heading = 0.0
        self.state, self.until, self.goal = "pause", 1.0, self.pos.copy()
        self.hidden = False
        self._t = world.data.time
        world.after_advance.append(self.update)
        self.update()

    # --- helpers -------------------------------------------------------------
    def _random_point(self):
        while True:
            p = self.rng.uniform(self.lo + 0.15, self.hi - 0.15)
            if self.zone.contains_point(p):
                return p

    def _dot_xy(self):
        if self.laser is None or not self.laser.is_on or self.laser.last_dot is None:
            return None
        p = self.laser.last_dot.point
        if p is None or not self.zone.contains_point(p[:2]):
            return None
        return p[:2]

    def _choose(self, now):
        r = self.rng.random()
        if r < 0.08 and self.hide_spots:
            self.state, self.goal = "to_hide", self.hide_spots[self.rng.integers(len(self.hide_spots))]
            self.until = now + 10
        elif r < 0.35:
            self.state, self.until = "pause", now + self.rng.uniform(0.8, 2.5)
        elif r < 0.5:
            self.state, self.goal, self.until = "dash", self._random_point(), now + 3
        else:
            self.state, self.goal, self.until = "wander", self._random_point(), now + 6

    # --- per-step ---------------------------------------------------------------
    def update(self) -> None:
        now = self.world.data.time
        dt, self._t = now - self._t, now
        dot = self._dot_xy()

        if self.state == "hidden":
            if now > self.until:  # come back out
                self.hidden, self.state, self.goal = False, "dash", self._random_point()
                self.until = now + 3
        elif dot is not None and self.state not in ("pounce", "to_hide"):
            dist = np.linalg.norm(dot - self.pos)
            if dist < 0.45:
                self.state, self.goal, self.until = "pounce", dot.copy(), now + 0.6
            elif dist < 1.0 and self.state != "stalk":
                self.state, self.until = "stalk", now + 4
            if self.state == "stalk":
                self.goal = dot.copy()
        if self.state not in ("hidden",) and (now > self.until or
                                              np.linalg.norm(self.goal - self.pos) < 0.03):
            if self.state == "to_hide":
                self.state, self.hidden, self.until = "hidden", True, now + self.rng.uniform(3, 7)
            elif self.state == "pause" or self.state in ("wander", "dash", "stalk", "pounce"):
                if self.state == "pounce":
                    self.state, self.until = "pause", now + self.rng.uniform(0.5, 1.5)
                else:
                    self._choose(now)

        if self.state in SPEED:
            step = self.goal - self.pos
            n = np.linalg.norm(step)
            if n > 1e-6:
                self.heading = math.atan2(step[1], step[0])
                self.pos = self.pos + step / n * min(n, SPEED[self.state] * dt)

        d = self.world.data
        z = -1.0 if self.hidden else 0.0
        d.mocap_pos[self._mocap] = [self.pos[0], self.pos[1], z]
        d.mocap_quat[self._mocap] = [math.cos(self.heading / 2), 0, 0, math.sin(self.heading / 2)]


class SimCatSource(CatSource):
    """Reads the fake cat's true position (sim only)."""

    def __init__(self, cat: FakeCat):
        self.cat = cat

    def get_cats(self, frame) -> list[Cat]:
        if self.cat.hidden:
            return []
        return [Cat(float(self.cat.pos[0]), float(self.cat.pos[1]), CAT_RADIUS)]
