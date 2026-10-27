# Stage 2 — where to grasp

**object point cloud + chosen gripper → GraspGenX → 6-DOF grasps → checked with the real gripper**

GraspGenX tells its model about a gripper through **twelve numbers**: the box swept between the
fingers when open and half-open. The first version of this stage did three things:
- gave it a 2-finger box and a square 4-finger box;
- measured "standoff": whether the object crosses the top of the box;
- found the 2-finger grasps reached the object and the 4-finger ones never did.

This README covers what was wrong with that, and the fix.

| | before (standoff, old box) | after (grasps closed with the real fingers) |
|---|---|---|
| 4-finger, on objects stage 1 sends to 4 fingers | 0% reach (10 Objaverse objects), 18% (10 YCB) | **45%** of top-10 grasps valid, **15 of 16** objects with a valid grasp |
| the same objects, the old 4-finger box | | 2% valid, 6 of 16 |
| 2-finger | 84% reach (10 Objaverse), 68% (10 YCB) | 76% valid on round objects, 30% on the rest; 44 of 50 objects with a valid grasp |

*The "84% vs 0%" quoted before came from the 10 Objaverse demo objects in stage 1. On the 10 YCB
objects the same metric gave 68% vs 18%.*

![results](results/grasp_check.png)

## What was wrong

1. **A box cannot say "four fingers".**
   - The released checkpoint conditions on the twelve numbers alone
     (`gripper_backbone: sweep_volume_v2`); the gripper type is not an input.
   - Every gripper it was trained on (parallel and revolute 2-finger, revolute 3-finger) closes
     along one axis. Between open and half-open its box narrows in x and keeps its y.
   - Our 4-finger box narrows in both: a gripper the model has never seen.
2. **The numbers did not match the gripper.**
   - `params_legacy.json` says the fingers sweep a 148 × 30 × 38 mm box.
   - Assembled from the STLs and opened until the fingertips are 148 mm apart, the fingers form a
     V instead: 66 mm apart at the root, 148 mm only at the tips. The tips reach 26 mm past the top
     of the box, and the fingers are 10 mm thick, not 30.
   - A 148 × 148 mm square reads as a huge hand, and the model backs off. The square box measured
     from the STLs (66 × 66 mm open) already does far better: 30% valid instead of 2%.
3. **The metric was the box's, not the gripper's.** Standoff only asked whether the object crossed
   the top of that box. It never asked whether a finger touches the object, whether the gripper
   collides with it, or whether the pinch can hold.

## The fix

**Check every grasp with the real gripper** (`finray.py`). The body, clamps and fingers are
assembled from the STLs (`assemble_gripper.py`). They are closed on the object one clamp-angle
step at a time, with FCL collision checks. A grasp is valid only if:
- nothing collides when fully open;
- every finger touches the object with its finger, not its clamp, before the fingers meet;
- with four fingers, the two pairs grip close enough together. All clamps turn off one rack, so
  the pair that grips first keeps being driven until the other grips. That extra travel must fit
  in the fin-ray's bending: 10 mm at the contact. The fin-ray sensing paper in `papers/` loads a
  finger of this size to 10 mm.
- each opposing pair squeezes along its contact line: the line between the two contacts lies
  inside both friction cones (μ = 0.6). Normals are averaged over the 5 mm patch a fin-ray presses.

**Plan four fingers as two pinches** (`four_finger.py`).
- Our 4-finger gripper is two of our 2-finger grippers on one approach axis, the second turned 90°.
  So a 4-finger grasp at pose T is good when the 2-finger grasp at T holds and so does the one at T
  turned 90°.
- GraspGenX's discriminator, given the 2-finger box it does understand, scores both pinches; the
  weaker one decides.
- Candidates come from GraspGenX itself: its default planner with the 2-finger box (diffusion
  samples plus its bounding-box sweep), and its grasps for the STL-sized square box. Each is turned
  0–75° about its approach axis in 15° steps.
- What the discriminator cannot know is that one rack drives both pairs. So the object must be
  about as wide under one pair as under the other, measured from the point cloud. The two widths
  may differ by at most 2 × 10 mm.
- Without this check, the planner pinches a soup can across its diameter with one pair and end to
  end with the other. That scored 0% on the soup can; with the check, 80%.

**Choose the 2-finger box by what it produces** (`tune_gripper_box.py`).
- No single box fits a V, so the box is treated as a description to calibrate.
- Eleven candidates (66–148 mm wide; the fingertip band or the whole finger; the old box among
  them) each planned grasps on 16 Objaverse objects from stage 1's training split.
- A candidate would replace the old box only if it beat it on the same objects (paired Wilcoxon,
  p < 0.05).
- None did. The best, essentially the wizard's 66 mm box, scored 0.21 against the old box's 0.19
  (p = 0.38). So the old box stays, and the 4-finger planner uses it too.

## Results

50 real-scanned objects: 26 YCB and 24 Google Scanned Objects, the same ones stage 1 is tested on.
For each method, the 10 grasps it scores highest are closed with the real fingers. Planning takes
about 1 s per object for 2 fingers and 2 s for 4, on the Mac CPU.

| method | stage 1 sends to 4 fingers (16 objects): top-10 valid | top-1 valid | any valid in top 100 | sends to 2 fingers (34): top-10 valid | any valid in top 100 |
|---|---|---|---|---|---|
| 2-finger, old box | 76% | 81% | 16 | 30% | 28 |
| 2-finger, wizard box from the STLs | 71% | 69% | 15 | 24% | 24 |
| 2-finger, old box + GraspMoE | 74% | 81% | 16 | 29% | 28 |
| 4-finger, old square box | 2% | 0% | 6 | 0% | 3 |
| 4-finger, square box from the STLs | 30% | 31% | 14 | 0% | 4 |
| 4-finger, two pinches, discriminator only | 37% | 44% | 15 | 12% | 14 |
| **4-finger, two pinches + one-rack check** | **45%** | **50%** | **15** | 13% | 14 |

![4-finger](results/grasp_check_4f.png)

*The best-scored 4-finger grasp, old square box (left) and the fixed planner (right), drawn with
the STLs. Each finger is closed to where it first touches. Each panel is drawn from the gripper's
side, so the object's orientation differs between panels.*

**What the results show.**
- **The fixed planner cages round objects as the gripper was designed to.** It approaches a
  cylinder along its axis and closes four fingers around the round cross-section. The old box
  approached from the side and stopped short. The only object stage 1 sends to 4 fingers that gets
  no valid 4-finger grasp is the 19 mm marker.
- **Four fingers also work on squat round objects the rule sends to two.** Examples: the ramekin
  (90%), the tuna can (100%), the coffee mug (90%) and the mug (70%). Their cross-section across
  the long axis is a flat rectangle, so the rule says 2-finger; seen from above they are circles.
  This is the rule's main blind spot.
- **The 2-finger gripper validly closes on round objects too** (76% of top-10 grasps). The
  geometric check cannot say whether two fingers would let a round object roll. That is the reason
  for the 4-finger configuration, and it is a stability question for physical trials or
  simulation.
- **Six objects get no valid 2-finger grasp:** the frying pan and five bowls. They are wider than
  the 148 mm opening, so a grasp has to pinch the thin rim wall, and on these grasps only one
  finger reaches it.
- **GraspMoE adds nothing for 2 fingers here.** It is GraspGenX's default planner (diffusion plus
  a bounding-box sweep), and it matches plain diffusion: 74% vs 76%.

## The whole pipeline

Stage 1's fine-tuned selector picks the gripper, and stage 2 plans it (`pipeline_check.py`). An
object is handled when the grasp the pipeline executes is valid: either the planner's top grasp,
or the best-scored valid grasp in its top 100 when `finray.py` is used as a filter.

| policy, 50 real-scanned objects | top grasp valid | filtered | picks 4-finger |
|---|---|---|---|
| old pipeline: the VLM always said 2-finger | 50% | 88% | 0% |
| old planners, with stage 1's rule picking correctly | 24% | 68% | 32% |
| **fixed selector + fixed planners** | **40%** | **86%** | 38% |
| always 4 fingers, fixed planner | 26% | 58% | 100% |

The old pipeline looked healthy only because its VLM never chose 4 fingers. A selector that did
choose them would have dropped it to 68%, because the old 4-finger planner failed on everything
sent to it. With both stages fixed, sending 38% of objects to four fingers costs almost nothing
geometrically (86% vs 88%). Whether those grasps hold better is the stability question above.

## Caveats

- **Geometry, not physics.** The check has no object weight, no table, and no deformation beyond
  the bending allowance. GraspGenX's own simulation extra is CUDA-only. The check rules grasps out;
  it does not prove the rest hold.
- **The opening angle is assumed.** The rack is not among the STLs, so the gripper is opened until
  the fingertips are 148 mm apart, the aperture the old box gave GraspGenX.
- **The 4-finger planner inherits the 2-finger model's blind spots.** It reuses GraspGenX's
  discriminator twice. It adds no knowledge of four fingers beyond the 90° composition, the
  one-rack width check and the final check.
- **16 objects need four fingers by the rule.** The interval on any 4-finger percentage here is
  wide (±20–25 points).

## Files and commands

```
scripts/finray.py              the gripper from the STLs, and the grasp check
scripts/four_finger.py         4-finger grasps as two 2-finger pinches, with the one-rack check
scripts/tune_gripper_box.py    choose the 2-finger box -> data/gripper/params_tuned.json, results/gripper_box_tuning.csv
scripts/evaluate_grasps.py     every method on the 50 objects -> results/grasp_check.csv (poses in data/grasp_eval/)
scripts/plot_grasps.py         results/grasp_check.png, results/grasp_check_4f.png
scripts/pipeline_check.py      stage 1's pick + stage 2's plan -> results/pipeline_check.csv
scripts/objects.py             YCB, Google Scanned and Objaverse meshes in metres
results/gripper_boxes.json     the old boxes and the ones measured from the STLs
stage2_grasp.ipynb, stage2_grasp_4finger.ipynb, scripts/graspgenx_selected_objects.py   the first version
```

```bash
G=/opt/anaconda3/envs/graspgenx/bin/python
$G stage2_grasp/scripts/tune_gripper_box.py       # ~40 min, resumes if stopped
$G stage2_grasp/scripts/evaluate_grasps.py        # ~15 min, resumes if stopped
$G stage2_grasp/scripts/plot_grasps.py
$G stage2_grasp/scripts/pipeline_check.py axis_lora
```
