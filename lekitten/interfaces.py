"""The four things every backend provides. Play, aim and safety code only uses these.

Implementations: backends/sim/ (MuJoCo) and backends/real/ (LeRobot SO-101, USB relay, webcam).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# LeRobot SO-101 joint names. Degrees, except gripper.pos: 0 (open) .. 100 (closed).
JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

FRAME_HEIGHT = 480
FRAME_WIDTH = 640


@dataclass(frozen=True)
class Cat:
    """A cat on the floor, in room coordinates (meters)."""
    x: float
    y: float
    radius: float


class Arm(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def read_joints(self) -> dict[str, float]:
        """Current joint positions, keyed by JOINTS names."""

    @abstractmethod
    def send_joints(self, joints: dict[str, float]) -> None:
        """Command joint positions (any subset of JOINTS)."""

    @abstractmethod
    def disconnect(self) -> None: ...


class OverheadCamera(ABC):
    @abstractmethod
    def read(self) -> np.ndarray:
        """RGB uint8 frame, shape (480, 640, 3)."""


class Laser(ABC):
    @abstractmethod
    def on(self) -> None: ...

    @abstractmethod
    def off(self) -> None: ...

    @property
    @abstractmethod
    def is_on(self) -> bool: ...


class CatSource(ABC):
    @abstractmethod
    def get_cats(self, frame: np.ndarray) -> list[Cat]:
        """Cats visible in this frame, in room coordinates."""
