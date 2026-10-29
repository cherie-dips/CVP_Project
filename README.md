# Grasping with a fin-ray gripper

Three stages.

```
stage 1   object views  +  gripper configurations (2- and 4-finger)
          ──► VLM ──►  which configuration to use

stage 2   that configuration  +  object point cloud
          ──► GraspGenX (4 fingers: two composed 2-finger pinches) ──►  6-DOF grasps
          ──► closed with the real gripper (STLs) ──►  the valid ones

stage 3   camera image of the finger  +  the SEN0704 tactile sensor
          ──► fine-tuned CNN ──►  grip force from the image alone
```

The gripper is ours: four fin-ray fingers in a cross, driven together by a rack.
**Remove two opposing fingers and it is the 2-finger gripper**: same body, same fingers, same
travel. That is the only choice stage 1 makes.

| | |
|---|---|
| [`stage1_gripper/`](stage1_gripper) | the VLM decision, its labels, fine-tuning |
| [`stage2_grasp/`](stage2_grasp) | GraspGenX, both configurations, and the real-gripper check |
| [`stage3_force/`](stage3_force) | tactile sensor calibration, marker tracking, force from the finger image |
| `data/gripper/` | the gripper itself: STLs, the swept-volume boxes (`params_*.json`), and `description.md`, the description the first selector was shown |
| `data/objaverse/`, `data/ycb/` | meshes and real-world scale factors (not tracked) |
| `papers/` | the papers this is built on |

## Results

| | before | after |
|---|---|---|
| stage 1, which gripper | 95.6% accuracy, **0% recall on 4-finger** (0 of 3): the same answer for every object | **94.6% balanced accuracy, 91.7% 4-finger recall** (44 of 48); 16 of 16 on real scans never trained on |
| stage 2, 4 fingers | **0% reach**: GraspGenX stops short of every object | **45%** of top-10 grasps valid on the objects stage 1 sends to 4 fingers; 15 of 16 get a valid grasp |
| stage 2, 2 fingers | 84% reach, by a metric that never touched the real fingers | 76% valid on round objects, 30% on the rest; 44 of 50 objects get a valid grasp |
| both stages together | worked only because stage 1 never chose 4 fingers | 86% of 50 real-scanned objects get a valid grasp, with 38% sent to 4 fingers |
| stage 2, physics simulation | | PyBullet, 8 Objaverse objects, top 10 grasps: **88%** held with 4 fingers, 85% with 2; the heavy round bottle 8/10 vs 1/10 |
| stage 3, force from the camera | from the 6 markers: 1.75 N error on unseen objects, no better than always predicting the average (1.70 N) | **1.14 N** with a ResNet-18 fine-tuned on the finger images; follows the force curve on every object (Spearman 0.60) but reads low above ~3 N |

**Stage 1 was reading the wrong output.** The VLM's generated answer was constant, but its logits
were not. Four fixes:
- read the decision from the logits, with a threshold set on validation data;
- train on that decision alone, with the rare class weighted up;
- show the model a view down the object's long axis, where the label is measured;
- evaluate on 362 + 50 objects instead of 68.

A linear probe on the frozen model already reaches 92%, so the information was in the image
features all along.

**Stage 2 was describing the wrong gripper.** GraspGenX knows a gripper only as a swept box, and
none of the grippers it was trained on closes along two axes. So four fingers are planned as two
2-finger pinches at 90°. GraspGenX's own discriminator scores each pinch, and a width check
enforces that one rack drives both pairs. Every grasp is then closed with the real gripper,
assembled from the STLs. That check also showed the old box did not match the fingers, which form
a V, and that the old "reach" numbers came from that box, not from the gripper.

Both are documented in the stage READMEs, with what the fixes do not show: the labels are our
rule, and the grasp check is geometry, not physics.

## Running it

GraspGenX and MLX cannot share an environment: GraspGenX pins numpy 1.26.4. Everything touching
the VLM runs in the `mlxvlm` env, everything else in `graspgenx`. GraspGenX itself is an upstream
clone, gitignored. The patch makes it run on CPU, because MPS is numerically wrong for this model.

```bash
git clone https://github.com/NVlabs/GraspGenX.git
cd GraspGenX && git apply ../data/gripper/graspgenx-mac-device.patch && pip install -e .
```

The commands for each stage are at the end of its README. On this 16 GB Mac, run the heavy jobs
(VLM training, GraspGenX evaluation) one at a time.
