# Moving to the real arm (planned, not done yet)

> **Status:** everything in this repo so far runs in simulation. This page is the plan for the real SO-101; the real-arm code exists and is tested in `--dry-run` mode (no hardware), but it hasn't been run on a robot yet. It will be updated with what actually happens.

Step by step, from a box of parts to the first supervised laser session. Copy each command, paste it into **Terminal**, press Return. Do the steps in order; if something prints an error, stop there.

**Laser rule for the whole day:** the laser stays **unplugged from the relay** until step 9. Steps 1–8 only move the arm.

---

## 1. Apple's command line tools (for git)

```bash
git --version
```

If a popup asks to install the **Command Line Tools**, click **Install** (about 10 minutes), then run `git --version` again. It should print `git version 2.x`.

## 2. Download the project

```bash
mkdir -p ~/robots && cd ~/robots && git clone https://github.com/rootyhub/LeKitten.git && cd LeKitten
```

## 3. Create the same Python sandbox as the work laptop

If Miniforge isn't installed yet, follow Step 2 of the Notion page first. Then:

```bash
conda env create -f environment.yml
conda activate lerobot
python -c "import mujoco, lerobot; print('MuJoCo', mujoco.__version__, '/ LeRobot ready')"
```

It should print `MuJoCo 4.0.0 / LeRobot ready`.

## 4. Find the ports

Plug the **follower arm's** driver board into the Mac by USB, and its power adapter into the wall. Then:

```bash
lerobot-find-port
```

It asks you to unplug the USB cable and press Return, and then tells you the arm's port (something like `/dev/tty.usbmodem5A7A0123451`). Open `config.yaml` and put it in:

```yaml
real:
  arm_port: /dev/tty.usbmodem5A7A0123451   # yours will differ
```

Plug in the **USB relay** (no laser attached to it yet). It's found automatically. If you ever get "Several possible relay ports", run `lerobot-find-port` again with the relay and put its port in `real.relay_port`.

Plug in the **overhead camera** and run:

```bash
lerobot-find-cameras opencv
```

Put the number of the Lemi camera in `real.camera_index`. If its picture comes out turned, set `real.camera_rotation_deg` to 90, 180 or 270.

## 5. Calibrate the arm

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=<your arm port> --robot.id=lekitten_follower
```

Follow its prompts: move the arm to the middle of its range, press Return, then move each joint through its full range. The id **must** be `lekitten_follower` (it's what `config.yaml` uses). The calibration file is saved by LeRobot on this Mac and never goes to GitHub.

## 6. First check on the real arm (laser off)

Clear the desk around the arm. Keep a hand near the power plug.

```bash
python -m lekitten.check --backend real
```

It prints the joints, then moves each one +20° and back, one at a time, and returns home. This is the **same program** you watched in the sim. If any joint moves the **opposite way** from the sim video, note which one for step 7.

## 7. Line up the real arm with the sim (joint_map)

Move the arm through 3 known poses (home, "reach forward", "turn left"). For each, it moves the real arm and the sim arm to the same numbers and you compare them side by side. Any joint that's mirrored gets `sign: -1`, and any constant gap gets an `offset_deg` in `config.yaml` under `real.joint_map`. Re-run step 6 until the real arm and the sim agree.

## 8. Lens correction and the camera's floor map

The Lemi camera has a fisheye lens, so its picture bends straight lines.

1. **Lens correction:** print a checkerboard, hold it in front of the camera in ~15 positions while a script saves frames, and it computes the lens correction once.
2. **Floor map (the "laser blink" trick):** the arm points at ~12 known floor spots from the sim. At each spot the laser blinks briefly (this is the one time the laser is used before step 9; nobody and no cats in the room). The camera finds the blink, and the pairs (camera pixel, floor position) give the map from camera pixels to the rug.

## 9. First supervised run

Only now plug the laser into the relay. **Cats in the room, you watching, hand on the power plug.**

```bash
python -m lekitten.play --backend real --duration 60
```

Until the cat detector exists (built next, on footage from this camera), the real program sees no cats, so the **laser stays off** by design and the arm only moves. The first sessions with the laser on happen once the detector is trained.

Every session ends with the dot stopping where it is, a short hold, then laser off. **Ctrl+C at any time turns the laser off** and stops the arm.

---

## If something goes wrong

| What you see | What to do |
| --- | --- |
| `real.arm_port is empty` | Do step 4 and save `config.yaml` |
| `No USB relay found` | Plug the relay in; check its light comes on |
| A joint moves the wrong way | Step 7 (`sign: -1` for that joint) |
| Arm jerks or overshoots | Lower `shared.arm.max_joint_speed_deg_s` in `config.yaml` (e.g. 30) |
| Anything unexpected | Unplug the arm's power first, then investigate. |

## Practise without hardware

Everything above can be rehearsed on any Mac without the arm:

```bash
python -m lekitten.check --backend real --dry-run
python -m lekitten.play  --backend real --dry-run --duration 10
```

They print exactly what they would send to the arm and the relay.
