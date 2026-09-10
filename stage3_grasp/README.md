# Stage 3 — object + mass + gripper → grasp points

Given the object and the gripper Stage 2 chose, say where to put the fingers.

Two separate parts:

1. **Propose** candidates from the object's shape — no learned generator
2. **Score** them with a small model trained on physics outcomes

Splitting it this way means every proposal is geometrically valid by construction, and the learned
part only has to rank.

## How proposals are made

- **2-finger** — find two opposing surface points closer together than the aperture, put the contact
  midway between them
- **4-finger** — centre on the object and approach along an axis it fits inside

Both reject grasps whose approach path is blocked.

Our gripper, measured from the STL files: finger **85 × 30 × 10 mm**, body **50.6 mm** square →
**148 mm** open aperture, **38 mm** reach.

## Results

**Proposals** on six real objects:

| object | size (mm) | 2-finger | 4-finger |
|---|---|---|---|
| tag | 157×60×91 | 200 grasps (5 mm) | 147 grasps (9 mm) |
| baseball cap | 132×73×158 | 200 (4 mm) | 0 |
| cloak | 138×40×211 | 200 (4 mm) | 0 |
| prawn | 108×102×233 | 200 (4 mm) | 0 |
| rifle | 17×60×329 | 200 (2 mm) | 0 |

Contacts land **2–9 mm from the surface**. The 4-finger gets zero on most objects because it genuinely
cannot grasp them — it needs the object to fit inside its 148 mm envelope on every axis. **That is a
feasibility check for Stage 2, for free.**

**Scoring**, trained on 89,512 physics-labelled grasps over 226 objects, tested on held-out objects:

| features | narrow gripper | wide gripper |
|---|---|---|
| geometry only | 0.607 | **0.587** |
| geometry + mass | **0.613** | 0.581 |

![Stage 3 results](results/stage3.png)

AUC is the chance the score ranks a successful grasp above a failed one; 0.5 is a coin flip. Mass moves
it by ±0.006 — nothing, and for the same reason as Stage 2: the training labels have no mass variation.

## A VLM can do this too

Same model and setup as Stage 2: **Qwen3-VL-2B**, LoRA on the Mac. Input is the object photo plus the
gripper Stage 2 chose; output is the contact points in image coordinates.

| | correct number of points | mean point error |
|---|---|---|
| zero-shot | 75% | 192 / 1000 |
| **fine-tuned** | **100%** | **112 / 1000** |

![Grasp points](results/vlm_grasp_points.png)

*Green = target, red = predicted, on the 8 test images.*

**Format compliance goes 75% → 100% and point error drops 42%.** Unlike Stage 2, fine-tuning genuinely
helps here — the task has a consistent geometric structure to learn rather than a lopsided class split
to memorise. Best case `screw` at 19/1000; worst `IMG_1862` at 187.

**One caveat.** We have no grasp-point annotations, so the targets were *derived*: take the model's own
object box, then apply the geometric rule (2-finger pinches across the short axis, 4-finger closes on
the centre). So this fine-tune **distils that rule into the model** — it collapses detect-then-compute
into one forward pass, which is useful, but it is not learning grasping from real trials.

Annotating where the fingers actually contacted in the 33 close-frames would turn this into real
supervision. It is about an hour of clicking.

## Run it

```bash
conda activate graspgenx
jupyter notebook stage3_grasp.ipynb
```
