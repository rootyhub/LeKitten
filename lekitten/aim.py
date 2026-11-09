"""Aiming: floor spot (x, y) in the room frame -> joint angles (LeRobot names, degrees).

    aimer = Aimer(cfg)                       # once (builds a small library of known-good poses)
    joints = aimer.aim_at(0.8, 1.2, current) # None if the spot can't be reached safely

Same code on both machines: it only uses the shared MuJoCo model of the arm (lekitten.kinematics).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import FLOOR_Z, ArmKinematics
from .model import room_cfg
from .zones import Zones

AIM_JOINTS = ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos")
TOLERANCE_M = 0.01      # accept a solution when the real dot is within 1 cm
FD_STEP_DEG = 0.25      # finite-difference step for the Jacobian


@dataclass
class AimResult:
    joints: dict[str, float]
    dot: np.ndarray          # where the dot actually lands (checked with the full laser cast)
    miss_m: float
    down_angle_deg: float
    surface: str


class Aimer:
    def __init__(self, cfg: dict, room: str | None = None, spot: str | None = None,
                 kin: ArmKinematics | None = None, library_size: int = 40_000, seed: int = 0):
        self.cfg = cfg
        self.kin = kin or ArmKinematics(cfg, room, spot)
        self.zones = Zones(room_cfg(cfg, room))
        las = cfg["shared"]["laser"]
        self.min_down = las["min_down_angle_deg"]
        lim = self.kin.jm.limits_deg()
        user = cfg["shared"]["arm"].get("joint_limits_deg") or {}
        self.lo = np.array([user.get(j, lim[j])[0] for j in AIM_JOINTS], float)
        self.hi = np.array([user.get(j, lim[j])[1] for j in AIM_JOINTS], float)
        self._build_library(library_size, seed)

    # --- helpers -----------------------------------------------------------
    def _pose(self, q, fixed) -> dict[str, float]:
        return {**dict(zip(AIM_JOINTS, map(float, q))), **fixed}

    def _check(self, joints: dict[str, float]):
        """Full safety check of one pose. Returns the Dot if it's a safe landing, else None."""
        self.kin.set_joints(joints)
        if self.kin.in_collision():
            return None
        dot = self.kin.dot()
        if dot.point is None or dot.down_angle_deg < self.min_down:
            return None
        if not (dot.on_floor or self.zones.surface_at(dot.point)):
            return None
        if self.zones.beam_crosses_no_aim(dot.origin, dot.point):
            return None
        return dot

    def _build_library(self, n: int, seed: int) -> None:
        """Known-good poses and where their dot lands; used as starting guesses."""
        rng = np.random.default_rng(seed)
        fixed = {"wrist_roll.pos": 0.0, "gripper.pos": 0.0}
        qs, pts = [], []
        for q in rng.uniform(self.lo, self.hi, size=(n, len(AIM_JOINTS))):
            dot = self._check(self._pose(q, fixed))
            if dot is not None:
                qs.append(q)
                pts.append(dot.point)
        self.lib_q = np.array(qs).reshape(-1, len(AIM_JOINTS))
        self.lib_p = np.array(pts).reshape(-1, 3)

    def target_point(self, x: float, y: float) -> np.ndarray | None:
        """The surface point at (x, y): floor, or couch seat, etc. (straight-down ray from 1.5 m)."""
        hit, _ = self.kin.beam.cast(self.kin.data, np.array([x, y, 1.5]), np.array([0.0, 0.0, -1.0]))
        return hit

    def _dot_on_plane(self, q, fixed, z) -> np.ndarray | None:
        """Where the beam crosses the horizontal plane at height z (smooth; used while solving)."""
        self.kin.set_joints(self._pose(q, fixed))
        o, v = self.kin.beam.ray(self.kin.data)
        if v[2] > -1e-3:
            return None
        return (o + v * (z - o[2]) / v[2])[:2]

    def _solve(self, q0, fixed, target, q_ref, iters: int = 40):
        q = np.clip(np.array(q0, float), self.lo, self.hi)
        # J is in meters per degree (~1e-2), so J^T J ~ 1e-4: keep damping well below that.
        # "Prefer small moves" is handled when choosing between solutions, not here.
        lam, w_ref = 1e-7, 0.0
        for _ in range(iters):
            p = self._dot_on_plane(q, fixed, target[2])
            if p is None:
                return None
            e = target[:2] - p
            if np.linalg.norm(e) < 1e-3:
                break
            J = np.zeros((2, len(q)))
            for i in range(len(q)):
                dq = q.copy()
                dq[i] += FD_STEP_DEG
                pi = self._dot_on_plane(dq, fixed, target[2])
                if pi is None:
                    return None
                J[:, i] = (pi - p) / FD_STEP_DEG
            # Damped least squares, gently pulled toward q_ref (small, smooth moves).
            A = J.T @ J + (lam + w_ref) * np.eye(len(q))
            step = np.linalg.solve(A, J.T @ e + w_ref * (q_ref - q))
            n = np.linalg.norm(step)
            if n > 10.0:
                step *= 10.0 / n
            q = np.clip(q + step, self.lo, self.hi)
        return q

    # --- public ------------------------------------------------------------
    def aim_at(self, x: float, y: float, current: dict[str, float] | None = None,
               n_seeds: int = 6) -> AimResult | None:
        """Joint angles that put the dot on (x, y), or None if it can't be done safely."""
        target = self.target_point(x, y)
        if target is None:
            return None
        surface = self.zones.surface_at(target)
        on_floor = target[2] < FLOOR_Z
        if surface is None and not on_floor:
            return None                      # furniture / couch while couch_play is off
        if any(v.contains(target).any() for v in self.zones.no_aim):
            return None
        fixed = {"wrist_roll.pos": (current or {}).get("wrist_roll.pos", 0.0),
                 "gripper.pos": (current or {}).get("gripper.pos", 0.0)}
        seeds = []
        if current is not None:
            seeds.append(np.array([current[j] for j in AIM_JOINTS]))
        if len(self.lib_p):
            d = np.linalg.norm(self.lib_p[:, :2] - target[:2], axis=1) + 2 * np.abs(self.lib_p[:, 2] - target[2])
            seeds += list(self.lib_q[np.argsort(d)[:n_seeds]])
        q_ref = seeds[0]
        best = None
        for s in seeds:
            q = self._solve(s, fixed, target, q_ref)
            if q is None:
                continue
            joints = self._pose(q, fixed)
            dot = self._check(joints)
            if dot is None:
                continue
            miss = float(np.linalg.norm(dot.point[:2] - target[:2]))
            if miss > TOLERANCE_M:
                continue
            move = float(np.abs(q - q_ref).sum())
            score = move - 0.5 * dot.down_angle_deg     # small moves, steeper beams
            if best is None or score < best[0]:
                best = (score, AimResult(joints, dot.point.copy(), miss, dot.down_angle_deg,
                                         surface.name if surface else "floor"))
        return best[1] if best else None
