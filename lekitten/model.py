"""Build the MuJoCo model for a room from config.yaml: room XML + collision shapes + arm + camera."""
from __future__ import annotations

import math
import warnings

import mujoco
import numpy as np

from lekitten.config import repo_path


def room_cfg(cfg: dict, room: str | None = None) -> dict:
    name = room or cfg["room"]
    return cfg["rooms"][name]


def yaw_quat(yaw_deg: float) -> list[float]:
    h = math.radians(yaw_deg) / 2
    return [math.cos(h), 0.0, 0.0, math.sin(h)]


def look_at_quat(eye, target, up=(0, 0, 1)) -> list[float]:
    """MuJoCo camera quaternion: camera looks along its -z, with +y up in the image."""
    eye, target, up = map(np.asarray, (eye, target, up))
    back = eye - target
    back = back / np.linalg.norm(back)          # camera +z
    right = np.cross(up, back)
    if np.linalg.norm(right) < 1e-6:            # looking straight down: pick +y as image up
        right = np.cross([0, 1, 0], back)
    right = right / np.linalg.norm(right)       # camera +x
    cam_up = np.cross(back, right)              # camera +y
    R = np.column_stack([right, cam_up, back])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q.tolist()


def build_model(cfg: dict, room: str | None = None, spot: str | None = None,
                visual: bool = True) -> mujoco.MjModel:
    """visual=True: the room's scene XML (scan, furniture looks) + a laser dot marker; for the sim.
    visual=False: only floor + collision shapes + arm; the geometric model aim/reach use on both machines."""
    rc = room_cfg(cfg, room)
    if visual:
        spec = mujoco.MjSpec.from_file(str(repo_path(rc["scene"])))
    else:
        spec = mujoco.MjSpec()
        floor = spec.worldbody.add_geom()
        floor.name, floor.type, floor.size = "floor", mujoco.mjtGeom.mjGEOM_PLANE, [0, 0, 0.05]
    arm = mujoco.MjSpec.from_file(str(repo_path(cfg["sim"]["arm_model"])))

    # Use the arm's physics settings (tuned for the SO-101).
    for k in ("integrator", "cone", "iterations", "ls_iterations", "impratio", "timestep"):
        setattr(spec.option, k, getattr(arm.option, k))

    # Invisible collision shapes for furniture (group 3 = hidden in the viewer by default).
    for ob in rc.get("obstacles", []):
        g = spec.worldbody.add_geom()
        g.name = f"obst_{ob['name']}"
        g.group = 3
        g.rgba = [1, 0.4, 0.2, 0.35]
        g.pos = ob["center"]
        if ob["type"] == "box":
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [s / 2 for s in ob["size"]]
            g.quat = yaw_quat(ob.get("yaw_deg", 0))
        else:
            g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            g.size = [ob["radius"], ob["height"] / 2, 0]

    # The arm, bolted to the chosen spot.
    s = rc["arm_spots"][spot or rc["arm_spot"]]
    frame = spec.worldbody.add_frame(pos=s["xyz"], quat=yaw_quat(s["yaw_deg"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame.attach_body(arm.body("base"), "", "")

    # Overhead camera at the configured pose.
    cc, cam = rc["camera"], cfg["shared"]["camera"]
    c = spec.worldbody.add_camera()
    c.name = "overhead"
    c.pos = cc["xyz"]
    c.quat = look_at_quat(cc["xyz"], cc["look_at"])
    c.fovy = cam["fov_deg"]

    if visual:  # fake cat (group 2: drawn, but not a laser target; safety keeps the dot off it)
        cat = spec.worldbody.add_body(name="cat", mocap=True, pos=[0, 0, -1])
        grey, dark = [0.62, 0.62, 0.6, 1], [0.25, 0.25, 0.25, 1]
        for name, gtype, size, pos, rgba in [
            ("cat_body", mujoco.mjtGeom.mjGEOM_ELLIPSOID, [0.2, 0.08, 0.08], [0, 0, 0.12], grey),
            ("cat_head", mujoco.mjtGeom.mjGEOM_SPHERE, [0.06, 0, 0], [0.22, 0, 0.18], grey),
            ("cat_ear_l", mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0, 0], [0.24, 0.035, 0.24], dark),
            ("cat_ear_r", mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0, 0], [0.24, -0.035, 0.24], dark),
            ("cat_tail", mujoco.mjtGeom.mjGEOM_CAPSULE, [0.015, 0.12, 0], [-0.3, 0, 0.18], dark),
        ]:
            g = cat.add_geom()
            g.name, g.type, g.size, g.pos, g.rgba = name, gtype, size, pos, rgba
            g.contype, g.conaffinity, g.group = 0, 0, 2
            if name == "cat_tail":
                g.quat = [0.92, 0, -0.38, 0]

    if visual:  # red laser dot, moved by the sim Laser; parked under the floor while off
        dot = spec.worldbody.add_body(name="laser_dot", mocap=True, pos=[0, 0, -1])
        g = dot.add_geom()
        g.name, g.type, g.size = "laser_dot", mujoco.mjtGeom.mjGEOM_SPHERE, [0.018, 0, 0]
        g.rgba, g.contype, g.conaffinity = [1, 0.05, 0.05, 1], 0, 0
        g.material = ""
        # Thin visible beam from the gripper to the dot (resized every update by the sim Laser).
        beam = spec.worldbody.add_body(name="laser_beam", mocap=True, pos=[0, 0, -1])
        g = beam.add_geom()
        g.name, g.type, g.size = "laser_beam", mujoco.mjtGeom.mjGEOM_CYLINDER, [0.002, 0.01, 0]
        g.rgba, g.contype, g.conaffinity = [1, 0.1, 0.1, 0.55], 0, 0

    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, 1920)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, 1080)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return spec.compile()
