"""Aiming tests. Use the public sample room so they run anywhere (no private scan needed)."""
import numpy as np
import pytest

from lekitten.aim import AIM_JOINTS, Aimer
from lekitten.config import load_config


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["room"] = "sample_room"
    return c


@pytest.fixture(scope="module")
def aimer(cfg):
    return Aimer(cfg, "sample_room", library_size=15_000)


def test_hits_toy_spot_within_1cm(aimer, cfg):
    x, y = cfg["rooms"]["sample_room"]["toy_spot"]
    r = aimer.aim_at(x, y)
    assert r is not None
    assert r.miss_m < 0.01


def test_most_of_play_zone_is_aimable(aimer):
    rng = np.random.default_rng(0)
    pts = rng.uniform([0.1, 0.1], [1.9, 1.4], size=(30, 2))
    ok = [aimer.aim_at(x, y) for x, y in pts]
    assert sum(r is not None for r in ok) >= 24
    assert max(r.miss_m for r in ok if r) < 0.01


def test_out_of_room_is_unreachable(aimer):
    assert aimer.aim_at(10.0, 10.0) is None


def test_couch_is_off_limits_without_couch_play(aimer):
    # Sample room has couch_play: false; the couch seat is a no-aim zone.
    assert aimer.aim_at(1.0, 2.5) is None


def test_solutions_respect_limits_and_beam_angle(aimer, cfg):
    min_down = cfg["shared"]["laser"]["min_down_angle_deg"]
    rng = np.random.default_rng(1)
    for x, y in rng.uniform([0.1, 0.1], [1.9, 1.4], size=(15, 2)):
        r = aimer.aim_at(x, y)
        if r is None:
            continue
        assert r.down_angle_deg >= min_down
        for i, j in enumerate(AIM_JOINTS):
            assert aimer.lo[i] - 1e-6 <= r.joints[j] <= aimer.hi[i] + 1e-6


def test_small_moves_from_current_pose(aimer):
    a = aimer.aim_at(1.0, 0.7)
    b = aimer.aim_at(1.05, 0.72, current=a.joints)
    assert b is not None
    move = max(abs(a.joints[j] - b.joints[j]) for j in AIM_JOINTS)
    assert move < 15.0  # a 5 cm nudge shouldn't swing the arm around
