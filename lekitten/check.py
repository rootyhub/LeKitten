"""First check on any arm: print joints, wiggle each one +20 and back, return home.

    mjpython -m lekitten.check --backend sim                 # with the 3D window (macOS)
    python   -m lekitten.check --backend sim --headless --video outputs/check.mp4
    python   -m lekitten.check --backend real                # the real arm (not built yet)
"""
from __future__ import annotations

import argparse

from .backend import add_backend_args, make_backend
from .config import load_config
from .interfaces import JOINTS
from .motion import HOME, move_to

WHAT = {
    "shoulder_pan.pos": "base turn",
    "shoulder_lift.pos": "shoulder (up/down)",
    "elbow_flex.pos": "elbow",
    "wrist_flex.pos": "wrist bend",
    "wrist_roll.pos": "wrist roll",
    "gripper.pos": "gripper (0 open, 100 closed)",
}


def show(joints: dict[str, float]) -> None:
    for j in JOINTS:
        print(f"  {j:<18} {joints[j]:8.1f}   {WHAT[j]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_backend_args(p)
    p.add_argument("--delta", type=float, default=20.0, help="How far to move each joint (degrees)")
    args = p.parse_args()
    speed = load_config(args.config)["shared"]["arm"]["max_joint_speed_deg_s"]

    b = make_backend(args)
    try:
        b.arm.connect()
        b.wait(0.5)
        print(f"[{b.name}] Current joints:")
        show(b.arm.read_joints())

        print("Going home...")
        move_to(b, HOME, speed)
        for j in JOINTS:
            here = b.arm.read_joints()
            print(f"Moving {j} ({WHAT[j]}) +{args.delta:g} and back")
            move_to(b, {**here, j: here[j] + args.delta}, speed)
            print(f"  reached {b.arm.read_joints()[j]:.1f}")
            move_to(b, here, speed)
        move_to(b, HOME, speed)
        print("Back home:")
        show(b.arm.read_joints())
        b.arm.disconnect()
    finally:
        b.close()


if __name__ == "__main__":
    main()
