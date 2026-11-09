"""Real Arm: the SO-101 follower through LeRobot 0.7 (lerobot.robots.so_follower.SO101Follower).

LeRobot already speaks our units (use_degrees=True): shoulder_pan.pos ... wrist_roll.pos in degrees,
gripper.pos 0..100. config.yaml real.joint_map (sign + offset_deg per joint) lines the real arm up
with the sim; it starts at sign 1 / offset 0 and is checked on arrival day with 3 known poses.
"""
from __future__ import annotations

from lekitten.interfaces import JOINTS, Arm


class RealArm(Arm):
    def __init__(self, cfg: dict, dry_run: bool = False):
        self.port = cfg["real"]["arm_port"]
        self.arm_id = cfg["real"]["arm_id"]
        self.map = cfg["real"]["joint_map"]
        self.dry_run = dry_run
        self.robot = None
        self._dry_state = {j: 0.0 for j in JOINTS}
        self._sends = 0

    # model (sim) degrees <-> robot degrees
    def _to_robot(self, j, v):
        m = self.map[j]
        return m["sign"] * v + m.get("offset_deg", 0.0)

    def _from_robot(self, j, v):
        m = self.map[j]
        return (v - m.get("offset_deg", 0.0)) / m["sign"]

    def connect(self) -> None:
        if self.dry_run:
            print(f"[dry-run] would connect SO101Follower(port={self.port!r}, id={self.arm_id!r}, use_degrees=True)")
            return
        if not self.port:
            raise SystemExit("real.arm_port is empty in config.yaml. Find it with: lerobot-find-port")
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        self.robot = SO101Follower(SO101FollowerConfig(port=self.port, id=self.arm_id, use_degrees=True))
        self.robot.connect()

    def read_joints(self) -> dict[str, float]:
        if self.dry_run:
            return dict(self._dry_state)
        obs = self.robot.get_observation()
        return {j: self._from_robot(j, float(obs[j])) for j in JOINTS}

    def send_joints(self, joints: dict[str, float]) -> None:
        action = {j: self._to_robot(j, v) for j, v in joints.items()}
        if self.dry_run:
            self._dry_state.update(joints)
            self._sends += 1
            if self._sends % 20 == 1:  # don't flood the terminal: about once a second at 20 Hz
                print("[dry-run] send_action " + ", ".join(f"{k.removesuffix('.pos')}={v:6.1f}"
                                                           for k, v in action.items()))
            return
        self.robot.send_action(action)

    def disconnect(self) -> None:
        if self.dry_run:
            print("[dry-run] would disconnect (torque off)")
            return
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None
