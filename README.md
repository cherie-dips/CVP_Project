# Grasping with a fin-ray gripper

Two stages.

```
stage 1   object point cloud / image  +  gripper configurations (2- and 4-finger)
          ──► LLM ──►  which configuration to use

stage 2   that configuration  +  object point cloud
          ──► GraspGenX ──►  6-DOF grasps
```

The gripper is ours: four fin-ray fingers in a cross, driven together by a rack.
**Remove two opposing fingers and it is the 2-finger gripper** — same body, same
fingers, same travel. That is the only choice stage 1 makes.

| | |
|---|---|
| [`stage1_gripper/`](stage1_gripper) | the LLM decision, its labels, and fine-tuning |
| [`stage2_grasp/`](stage2_grasp) | GraspGenX, both configurations |
| `data/gripper/` | the gripper itself: STLs, the 12 swept-volume numbers, and `description.md` — the file stage 1 shows the LLM |
| `data/objaverse/` | meshes, real-world scale factors, names, split lists |
| `data/ycb/` | YCB meshes for stage 2 (downloaded on first run) |
| `papers/` | the five papers this is built on |

## Results so far

| | |
|---|---|
| stage 1, zero-shot | 94.1% accuracy — and **0% recall** on the 4-finger class |
| stage 1, fine-tuned on 26 balanced examples | 95.6%, still **0% recall** |
| stage 2, 2-finger | 84% of grasps reach the object, median standoff −10 mm |
| stage 2, 4-finger | **0% reach**, median standoff +28 mm |

Two findings behind those numbers:

**GraspGenX cannot plan for the 4-finger gripper.** Its gripper input is a swept-volume
box; nothing in those twelve numbers says "four fingers". The second pair makes the box
square, which reads as a bulkier gripper, so the model backs the pose off and returns
confident grasps that close on empty air.

**The LLM is not deciding.** It answers the same configuration for every object, stably
under reordering the options — and because round objects are about one in five, that
scores 94% while being useless. Two different models fail this way in opposite
directions, so the bottleneck is the task as posed, not the model.

Both are documented in the stage READMEs.

## Running it

GraspGenX and MLX cannot share an environment (GraspGenX pins numpy 1.26.4), so the
notebooks shell out to the `mlxvlm` env for anything touching the VLM. GraspGenX itself
is an upstream clone, gitignored; the patch makes it run on CPU — MPS is numerically
wrong for this model.

```bash
git clone https://github.com/NVlabs/GraspGenX.git
cd GraspGenX && git apply ../data/gripper/graspgenx-mac-device.patch && pip install -e .
```
