# Stage 2 — object + mass → which gripper

Choose between our **2-finger** and **4-finger** fin-ray configurations.

Both are the same gripper: the 2-finger is the 4-finger with two fingers removed. Identical fingers,
identical stiffness — only the number of contacts changes.

## The data problem

We have no labelled trials of our own, so we train on the closest public stand-in: GraspGen's
**narrow Franka (80 mm)** vs **wide Robotiq (140 mm)**, 8,454 objects with physics-simulated outcomes.
Different grippers, same kind of decision.

## Results

**Regret** — how much grasp success is given up by not choosing the better gripper. Lower is better.

| strategy | regret | accuracy |
|---|---|---|
| always wide | 0.2851 | 0.210 |
| shape only | 0.0516 | 0.790 |
| **shape + mass** | **0.0500** | 0.774 |
| always narrow | 0.0505 | 0.790 |

![Stage 2 results](results/stage2.png)

**Nothing beats always picking one gripper**, and **mass adds nothing** (0.0516 → 0.0500).

Two reasons, and only one is fixable:

1. **There is almost nothing to win.** The oracle recovers just **4.8 points** over the best fixed
   choice, because the narrow gripper wins 73% of the time.
2. **Mass cannot help on this data.** GraspGen's simulation used **one fixed density and friction for
   every object**, so the labels contain no mass information at all. This is not evidence against the
   pipeline — it is evidence that the proxy cannot test it.

## Our own hardware

33 grasp clips, configurations read off the video frames (provisional — 9 high confidence, 9 medium,
15 low).

| | |
|---|---|
| 4-finger clips | 25 |
| 2-finger clips | 6 |
| **objects tried in BOTH configurations** | **3** — banana, marker, mug |

Those three are the only data that could actually train this stage. **More paired trials is the single
most useful thing to collect**, especially the 2-finger versions of egg, pinkball, rubik and wheel,
which already have 4-finger clips.

## Run it

```bash
conda activate graspgenx
jupyter notebook stage2_gripper.ipynb
```
