# LeKitten

Training an SO-ARM101 to play with my kittens using a laser pointer.

The arm points a Class 2 (under 1 mW) laser at the floor, leads the cats around, never puts the dot on a cat or on the couch, and ends every session calmly: the dot stops, holds a moment, and the laser turns off.

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
mjpython -m lekitten.check --backend sim                 # each joint wiggles in the 3D window (macOS needs mjpython)
mjpython -m lekitten.play  --backend sim                 # the full game with the fake cat
python   -m lekitten.play  --backend sim --headless --duration 120 \
         --video outputs/play_3d.mp4 --camera-video outputs/play_overhead.mp4
python   -m lekitten.tools.reach_map                     # where can the dot go from each arm spot?
python   -m lekitten.tools.aim_test                      # how accurately does it aim?
python   -m lekitten.check --backend real --dry-run      # rehearse the real arm without hardware
pytest                                                   # safety, aiming and zone tests
```

On the real arm, follow [docs/ARRIVAL_DAY.md](docs/ARRIVAL_DAY.md).

## Safety

Class 2 laser (under 1 mW), plus rules in `lekitten/safety.py` that every command goes through, in sim and real:
the beam always points down, never at or across a cat, never through the couch where people sit;
the laser is off by default, off when no cat is seen, off on any error or exit, and every session ends with the dot holding still, then laser off.

## Credits

- Built with [LeRobot](https://github.com/huggingface/lerobot) and [MuJoCo](https://mujoco.org).
- SO-101 model (`sim/models/so101/`): [The Robot Studio SO101 MJCF](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101) from MuJoCo Menagerie (commit `98a70c5`), Apache License 2.0. Its LICENSE is kept alongside the model.
