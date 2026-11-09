"""Let a trained policy (the student) play, with the same safety rules as the hand-written player.

    python -m lekitten.policy_play --policy outputs/train/t4/checkpoints/last/pretrained_model \\
        --headless --duration 45 --video outputs/student_3d.mp4

The policy sees what it saw in training (desk + wrist camera at 320x240, joint angles) and proposes
joint targets. Every proposal goes through safety.py (speed and joint limits, beam down, dot never on
or across a cat, never through the couch); the laser guard alone decides whether the laser is on.
After the play time, the session ends the same way as always: dot to the toy, hold, laser off.
Sim only for now (the real arm needs its wrist camera wired in first, in T7).
"""
from __future__ import annotations

import argparse
import math

import numpy as np
from PIL import Image

from .aim import Aimer
from .backend import make_backend
from .config import load_config
from .interfaces import JOINTS
from .model import room_cfg
from .play import build_parser, step_toward
from .safety import LaserGuard, Safety

FPS = 30


def reason_group(why: str) -> str:
    """Group refusal reasons for reporting ("beam only -4 deg..." and "...7 deg..." are the same rule)."""
    if why.startswith("beam only"):
        return "beam not pointing down enough"
    if why.startswith("beam crosses no-aim"):
        return "beam crosses the couch (people)"
    return why


class Student:
    def __init__(self, path: str, device: str = "mps", ensemble: float | None = None):
        """ensemble: if set (e.g. 0.01), use ACT's temporal ensembling: re-plan every frame and
        average the overlapping plans for smoother motion (no retraining needed)."""
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        self.torch = torch
        self.policy = ACTPolicy.from_pretrained(path)
        self.policy.to(device)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=path)
        if ensemble is not None:
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
            self.policy.config.temporal_ensemble_coeff = float(ensemble)
            self.policy.config.n_action_steps = 1
            self.policy.temporal_ensembler = ACTTemporalEnsembler(float(ensemble), self.policy.config.chunk_size)
        self.size = tuple(self.policy.config.input_features["observation.images.overhead"].shape[1:])  # (H, W)

    def reset(self) -> None:
        self.policy.reset()  # clears the queued action chunk between episodes

    def _img(self, rgb: np.ndarray):
        h, w = self.size
        if rgb.shape[:2] != (h, w):
            rgb = np.asarray(Image.fromarray(rgb).resize((w, h), Image.BOX))
        return self.torch.from_numpy(np.ascontiguousarray(rgb).copy()).permute(2, 0, 1).float() / 255.0

    def act(self, state: dict, overhead: np.ndarray, wrist: np.ndarray) -> dict[str, float]:
        t = self.torch
        obs = {"observation.state": t.tensor([state[j] for j in JOINTS], dtype=t.float32),
               "observation.images.overhead": self._img(overhead),
               "observation.images.wrist": self._img(wrist),
               "task": "Lead the cat around the rug with the laser dot, then land it on the toy."}
        with t.inference_mode():
            a = self.post(self.policy.select_action(self.pre(obs)))
        a = a.squeeze(0).float().cpu().numpy()
        return {j: float(v) for j, v in zip(JOINTS, a)}


def run_student(student: Student, args: argparse.Namespace, aimer: Aimer, start_pose: dict | None = None,
                on_tick=None, verbose: bool = True) -> dict:
    from backends.sim.camera import SimCamera

    from .motion import move_to
    log = print if verbose else (lambda *a, **k: None)
    cfg = load_config(args.config)
    cfg["room"] = args.room or cfg["room"]
    s, play = cfg["shared"]["safety"], cfg["shared"]["play"]
    duration = args.duration or play["session_s"]
    zones = aimer.zones
    safety = Safety(cfg, zones, aimer.kin)
    b = make_backend(args)
    cam = cfg["shared"]["camera"]
    wrist = SimCamera(b.world, "wrist_cam", width=cam["width"], height=cam["height"])
    dt = 1.0 / FPS
    stats = {"ticks": 0, "laser_on": 0, "blocked": {}, "ended_on_toy": False, "play_ticks": 0,
             "play_laser_on": 0, "cat_dot": []}
    student.reset()
    try:
        b.arm.connect()
        if start_pose:
            move_to(b, start_pose, cfg["shared"]["arm"]["max_joint_speed_deg_s"])
        with LaserGuard(b.laser, no_cat_timeout_s=s["no_cat_timeout_s"],
                        max_session_s=duration + s["toy_hold_s"] + 20,
                        clock=lambda: b.world.data.time) as guard:
            current = b.arm.read_joints()
            dot = None
            hold_until = None
            while True:
                now = guard.elapsed
                state = b.arm.read_joints()
                frame = b.camera.read()
                cats = b.cats.get_cats(frame)
                guard.saw_cats(cats)
                if now < duration:  # the student plays
                    proposal = student.act(state, frame, wrist.read())
                    target = safety.limit_speed(current, proposal, dt)
                    verdict = safety.pose_ok(target, cats)
                    b.arm.send_joints(target)
                    current = target
                    why = guard.set(True, verdict)
                    stats["play_ticks"] += 1
                    stats["play_laser_on"] += b.laser.is_on
                    if why:
                        key = reason_group(why)
                        stats["blocked"][key] = stats["blocked"].get(key, 0) + 1
                else:  # end of session, same as always: dot to the toy, hold, laser off
                    if dot is None:
                        aimer.kin.set_joints(current)
                        d = aimer.kin.dot()
                        dot = d.point[:2] if d.point is not None else zones.toy.copy()
                    dot = step_toward(dot, zones.toy, play["dot_speed_m_s"] * dt)
                    aim = aimer.aim_at(dot[0], dot[1], current)
                    verdict = None
                    if aim is not None:
                        target = safety.limit_speed(current, aim.joints, dt)
                        verdict = safety.pose_ok(target, cats)
                        b.arm.send_joints(target)
                        current = target
                    guard.set(aim is not None, verdict)
                    if np.linalg.norm(dot - zones.toy) < 0.01:
                        hold_until = hold_until or now + s["toy_hold_s"]
                        if now >= hold_until:
                            guard.set(False)
                            stats["ended_on_toy"] = True
                            log("Landed on the toy, held, laser off. Session done.")
                            break
                    if now > duration + 15:
                        log("Couldn't reach the toy in time; laser off.")
                        break
                stats["ticks"] += 1
                stats["laser_on"] += b.laser.is_on
                if now < duration and b.laser.is_on and cats and b.laser.last_dot is not None \
                        and b.laser.last_dot.point is not None:
                    p = b.laser.last_dot.point
                    stats["cat_dot"].append(min(math.hypot(p[0] - c.x, p[1] - c.y) for c in cats))
                if on_tick:
                    on_tick(b, state, frame, dict(current))
                b.wait(dt)
        b.arm.disconnect()
    finally:
        b.close()
    on = stats["play_laser_on"] / max(1, stats["play_ticks"])
    log(f"Laser on {on:.0%} of the play time.")
    if stats["blocked"]:
        log("Laser held off because:", ", ".join(f"{k} ({v} ticks)" for k, v in
                                                 sorted(stats["blocked"].items(), key=lambda kv: -kv[1])))
    return stats


def main() -> None:
    p = build_parser()
    p.add_argument("--policy", required=True, help="Path to a checkpoint's pretrained_model folder")
    p.add_argument("--ensemble", type=float, default=None,
                   help="Temporal ensembling coefficient (e.g. 0.01) for smoother motion")
    p.set_defaults(room="living_room", headless=False)
    args = p.parse_args()
    args.backend = "sim"
    cfg = load_config(args.config)
    cfg["room"] = args.room
    print("Loading the student and preparing aim...")
    student = Student(args.policy, ensemble=args.ensemble)
    aimer = Aimer(cfg)
    args.cat_seed = args.seed
    run_student(student, args, aimer)


if __name__ == "__main__":
    main()
