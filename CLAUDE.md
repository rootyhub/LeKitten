# LeKitten — rules for Claude Code

## What this is
An SO-101 robot arm that points a Class 2 (<1 mW) laser at the floor to play with two cats, plus an open method anyone can reproduce. The owner is not a programmer: explain what you do in plain words, keep steps small, and run things to show they work.

## Machines
- Work laptop (macOS, Apple Silicon): MuJoCo simulation only. No robot hardware.
- Home Mac (macOS, Apple Silicon): the real arm, USB relay laser, cameras. Gets code via git pull.
- Both use the conda env "lerobot" (Python 3.12) defined in environment.yml. Keep versions identical.

## The one rule: one pipeline, two backends
All play, aim and safety code uses only these interfaces (lekitten/interfaces.py):
- Arm: connect(), read_joints() -> dict, send_joints(dict), disconnect()
- OverheadCamera: read() -> RGB uint8 array, 480 high x 640 wide
- Laser: on(), off(), is_on
- CatSource: get_cats(frame) -> list of Cat(x, y, radius) in room coordinates (meters)
Implementations live in backends/sim/ (MuJoCo) and backends/real/ (LeRobot SO-101 follower + LCUS-1 USB relay + OpenCV webcam). Pick with --backend sim|real (default from config.yaml).
- MuJoCo physics, rendering and the fake cat live only in backends/sim/ and sim/.
- Hardware libraries (serial, LeRobot robot classes, webcam) live only in backends/real/.
- Kinematics (lekitten/aim.py) may use the MuJoCo model of the arm on BOTH machines; it is the shared geometric model of the SO-101.
- Every program is run the same way in sim and real, e.g. python -m lekitten.play --backend sim|real. Real-backend code must also support --dry-run (no hardware, prints what it would send).

## Shared conventions (must match the real robot)
- Joint names and units = LeRobot SO-101: shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos in degrees; gripper.pos 0 (open) to 100 (closed). The sim backend converts to/from MuJoCo radians using joint_map (sign + offset per joint) in config.yaml. The real joint_map is verified on arrival day.
- Room frame: meters, z up, origin at one floor corner of the play zone. Arm base pose, camera pose, play zone and no-aim zones are all defined once in config.yaml in this frame and used by both backends.
- Overhead camera: 640x480 RGB at 30 fps in both backends; the sim camera's field of view matches the real camera in config.yaml.
- Laser: a ray from the gripper's fixed jaw along the gripper's forward axis, with the mount offset in config.yaml. The dot is where the ray first hits the floor or furniture.
- Before writing any real-backend code, check the installed lerobot version's actual class and config names for the SO-101 follower (they have changed between versions).

## Safety rules (lekitten/safety.py, used by both backends, covered by tests in tests/)
- The beam always points below horizontal by at least the configured angle.
- The dot is never inside any cat's radius plus a margin, and never inside a no-aim zone (couch, where people sit).
- Laser defaults to OFF. It turns off when no cat is seen for N seconds, on any error, on exit, and after the max session length.
- Joint speed limits and joint limits inside the calibrated range.
- Every session ends calmly: the dot stops where it is, holds briefly (end_hold_s), then laser off.

## Privacy: this repo is PUBLIC
Never commit: sim/private_scan/ (the scan and photos of the owner's home), calibration/ (real-arm calibration), outputs/, tokens or keys or .env files, videos (*.mp4, *.mov), datasets. Keep them in .gitignore. sim/scenes/sample_room.xml is the public stand-in so others can run the sim without a scan.

## Working style
- One step per session. Run what you build and show the result (3D viewer, or images/videos saved to outputs/).
- On macOS, scripts that keep the interactive viewer open (launch_passive) must run with mjpython instead of python. Tell the owner which command to use.
- Small commits with clear messages. Only commit and push when the owner asks.
- Credit third-party assets (MuJoCo Menagerie SO-101 model, with its license) in the README.
