# LeKitten

Training an SO-ARM101 to play with my kittens using a laser pointer.

The arm points a Class 2 (under 1 mW) laser at the floor, leads the cats around, never puts the dot on a cat or on the couch, and ends every session on their toy. Everything is built and tested in a MuJoCo simulation first, then runs unchanged on the real arm.

## How it's organised

| Folder | What's in it |
| --- | --- |
| `lekitten/` | The shared brain: interfaces, aiming, safety rules, play behaviours, tools |
| `backends/sim/` | MuJoCo versions of the arm, overhead camera, laser and a fake cat |
| `backends/real/` | The real SO-101 (via LeRobot), USB relay laser and webcam |
| `sim/models/` | The SO-101 3D model |
| `sim/scenes/` | Rooms to simulate. `sample_room.xml` is public; bring your own scan for the rest |
| `tests/` | Automatic checks, mostly for the safety rules |
| `docs/` | Write-ups and the arrival-day guide |

Every program runs the same way in both worlds:

```bash
python -m lekitten.play --backend sim    # simulation
python -m lekitten.play --backend real   # the real arm
```

## Setup

```bash
conda env create -f environment.yml
conda activate lerobot
```

## Your own room

Put a LiDAR scan of your room (OBJ or GLB) in `sim/private_scan/`. That folder is git-ignored, so your home never ends up on GitHub. Then:

```bash
python -m lekitten.tools.import_scan "sim/private_scan/<scan>.obj" --origin X Y   # X Y = play-zone corner
python -m lekitten.tools.show_room --room living_room --snapshot
```

and fill in the `living_room` entry in `config.yaml`. Without a scan, the sim uses `sim/scenes/sample_room.xml`.

## Try it

```bash
mjpython -m lekitten.check --backend sim     # watch each joint wiggle in the 3D window (macOS needs mjpython)
python -m lekitten.check --backend sim --headless --video outputs/check.mp4
```

## Credits

- Built with [LeRobot](https://github.com/huggingface/lerobot) and [MuJoCo](https://mujoco.org).
- SO-101 model (`sim/models/so101/`): [The Robot Studio SO101 MJCF](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101) from MuJoCo Menagerie (commit `98a70c5`), Apache License 2.0. Its LICENSE is kept alongside the model.
