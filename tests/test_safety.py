"""Safety tests: try hard to break each rule. Use the public sample room (runs anywhere)."""
import numpy as np
import pytest

from lekitten.aim import AIM_JOINTS, Aimer
from lekitten.config import load_config
from lekitten.interfaces import JOINTS, Cat, Laser
from lekitten.safety import LaserGuard, Safety, Verdict


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["room"] = "sample_room"
    return c


@pytest.fixture(scope="module")
def aimer(cfg):
    return Aimer(cfg, "sample_room", library_size=15_000)


@pytest.fixture(scope="module")
def safety(cfg, aimer):
    return Safety(cfg, aimer.zones, aimer.kin)


class FakeLaser(Laser):
    def __init__(self):
        self._on = False
        self.history = []

    def on(self):
        self._on = True
        self.history.append("on")

    def off(self):
        self._on = False
        self.history.append("off")

    @property
    def is_on(self):
        return self._on


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


# --- Rule 1: beam always below horizontal ---------------------------------------
def test_level_or_upward_beam_is_refused(safety):
    rng = np.random.default_rng(0)
    lim = safety.limits
    refused = 0
    for q in rng.uniform([lim[j][0] for j in AIM_JOINTS], [lim[j][1] for j in AIM_JOINTS], size=(3000, 4)):
        joints = {**dict(zip(AIM_JOINTS, q)), "wrist_roll.pos": 0.0, "gripper.pos": 0.0}
        v = safety.pose_ok(joints, [])
        safety.kin.set_joints(joints)
        down = safety.kin.dot().down_angle_deg
        if down < safety.min_down:
            assert not v.ok, f"accepted a beam only {down:.1f} deg below horizontal"
            refused += 1
    assert refused > 100  # the random poses really did include flat/upward beams


# --- Rule 2: never on or through a cat, never in a no-aim zone -------------------------
def test_dot_on_cat_is_refused(safety, aimer):
    r = aimer.aim_at(1.0, 0.7)
    cat_on_dot = Cat(float(r.dot[0]), float(r.dot[1]), 0.2)
    assert not safety.pose_ok(r.joints, [cat_on_dot]).ok


def test_dot_just_outside_margin_is_allowed(safety, aimer):
    r = aimer.aim_at(1.0, 0.7)
    d = 0.2 + safety.cat_margin + 0.05
    # Put the cat beside the dot, away from the arm so the beam doesn't pass over it.
    cat = Cat(float(r.dot[0]), float(r.dot[1] - d), 0.2)
    assert safety.target_ok(r.dot[0], r.dot[1], [cat]).ok


def test_cat_margin_boundary(safety):
    cat = Cat(0.0, 0.0, 0.2)
    inside = 0.2 + safety.cat_margin - 0.001
    assert not safety.target_ok(inside, 0.0, [cat]).ok
    assert safety.target_ok(inside + 0.01, 0.0, [cat]).ok


def test_beam_over_cat_is_refused(safety, aimer):
    # Dot far from the cat, but the cat sits under the beam between arm and dot.
    r = aimer.aim_at(0.6, 0.75)
    safety.kin.set_joints(r.joints)
    dot = safety.kin.dot()
    # Put the cat 0.6 m back along the beam (toward the arm), where the beam is low.
    back = (dot.origin - dot.point)[:2]
    pos = dot.point[:2] + 0.6 * back / np.linalg.norm(back)
    cat = Cat(float(pos[0]), float(pos[1]), 0.2)
    assert safety.target_ok(dot.point[0], dot.point[1], [cat]).ok  # the dot itself is far enough away
    assert not safety.pose_ok(r.joints, [cat]).ok                  # but the beam passes over the cat


def test_couch_never_accepted_without_couch_play(safety, aimer):
    # Sweep many poses; none may put the dot on the couch (sample room: couch_play false).
    rng = np.random.default_rng(2)
    lim = safety.limits
    for q in rng.uniform([lim[j][0] for j in AIM_JOINTS], [lim[j][1] for j in AIM_JOINTS], size=(4000, 4)):
        joints = {**dict(zip(AIM_JOINTS, q)), "wrist_roll.pos": 0.0, "gripper.pos": 0.0}
        if safety.pose_ok(joints, []).ok:
            p = safety.kin.dot().point
            assert not (0.0 <= p[0] <= 2.0 and 2.0 <= p[1] <= 3.0), f"dot accepted on couch at {p}"


# --- Rule 3: laser guard -------------------------------------------------------
def make_guard(**kw):
    clock = Clock()
    laser = FakeLaser()
    g = LaserGuard(laser, no_cat_timeout_s=kw.get("timeout", 10), max_session_s=kw.get("session", 300), clock=clock)
    return g, laser, clock


def test_laser_starts_off_and_needs_a_cat():
    g, laser, clock = make_guard()
    with g:
        assert not laser.is_on
        assert g.set(True) == "no cat seen recently"
        assert not laser.is_on
        g.saw_cats([Cat(0, 0, 0.2)])
        assert g.set(True) == ""
        assert laser.is_on


def test_laser_off_after_no_cat_timeout():
    g, laser, clock = make_guard(timeout=10)
    with g:
        g.saw_cats([Cat(0, 0, 0.2)])
        g.set(True)
        clock.t = 10.5
        g.saw_cats([])
        g.set(True)
        assert not laser.is_on


def test_laser_off_after_max_session():
    g, laser, clock = make_guard(session=300)
    with g:
        g.saw_cats([Cat(0, 0, 0.2)])
        g.set(True)
        clock.t = 300.0
        g.saw_cats([Cat(0, 0, 0.2)])
        assert g.set(True) == "session over"
        assert not laser.is_on


def test_laser_off_when_safety_says_no():
    g, laser, clock = make_guard()
    with g:
        g.saw_cats([Cat(0, 0, 0.2)])
        g.set(True)
        g.set(True, Verdict(False, "too close to a cat"))
        assert not laser.is_on


def test_laser_off_on_error_and_exit():
    g, laser, clock = make_guard()
    with pytest.raises(RuntimeError):
        with g:
            g.saw_cats([Cat(0, 0, 0.2)])
            g.set(True)
            assert laser.is_on
            raise RuntimeError("motor fault")
    assert not laser.is_on
    assert laser.history[-1] == "off"


# --- Rule 4: joint limits and speed ------------------------------------------------
def test_targets_outside_limits_are_refused(safety):
    lo, hi = safety.limits["elbow_flex.pos"]
    joints = {j: 0.0 for j in JOINTS}
    joints["elbow_flex.pos"] = hi + 5
    assert not safety.pose_ok(joints, []).ok


def test_speed_limit_caps_every_joint(safety):
    cur = {j: 0.0 for j in JOINTS}
    tgt = {j: 90.0 for j in JOINTS}
    out = safety.limit_speed(cur, tgt, dt=0.05)
    step = safety.max_speed * 0.05
    assert all(abs(out[j] - cur[j]) <= step + 1e-9 for j in JOINTS)


def test_speed_limit_also_clamps_to_limits(safety):
    cur = {j: safety.limits[j][1] for j in JOINTS}
    tgt = {j: safety.limits[j][1] + 50 for j in JOINTS}
    out = safety.limit_speed(cur, tgt, dt=1.0)
    assert all(out[j] <= safety.limits[j][1] + 1e-9 for j in JOINTS)
