# Step 2 — Which gripper?

Choose between our **2-finger** and **4-finger** fin-ray grippers.

Both are the same gripper: the 2-finger version is the 4-finger one with two fingers removed. Same
material, same finger length, same softness — only the number of contacts changes.

This is the part of the project with no existing solution. Steps 1 and 3 use published models; this
one we build.

## Our pipeline

```
  object
    │
    ├─ what it is ──► CLIP text encoder ──► 512 numbers
    │
    └─ how big it is ──► size features
                          │
                          ▼
                   small neural net ──► which gripper wins, and by how much
```

Trained on GraspGen's two grippers as a stand-in for ours — a narrow Franka (80 mm) and a wide
Robotiq (140 mm) — over **8,454 objects** that both grippers attempted. Whole product categories are
held out, so the model is tested on object kinds it never saw.

## Results

Scored with **regret**: how much grasp success is given up by not choosing the better gripper.
Lower is better.

| strategy | regret | accuracy |
|---|---|---|
| always robotiq | 0.2599 | 0.258 |
| LLM, no examples | 0.1767 | 0.456 |
| LLM, 16 examples | 0.1686 | 0.480 |
| random | 0.1557 | 0.493 |
| **our model** | **0.0470** | 0.748 |
| always franka | 0.0474 | 0.742 |
| oracle | 0.0000 | 1.000 |

**Nothing beats simply always picking one gripper.** Our model matches it; the language model is
three times worse.

Three things explain it:

1. **There was almost nothing to win.** The oracle recovers only 4.8 points over the best fixed
   choice, and 20% of objects hold 98% of that.
2. **The model does learn the right thing** — predicted and true success gaps correlate at
   **r = +0.41**. It ranks objects correctly but rarely picks the wide gripper, which wins only 27%
   of the time.
3. **The mechanism is sensible.** Thin flat objects (dollar, map, wrench) favour the narrow gripper;
   round bulky ones (seashell, pumpkin, teapot) favour the wide one.

![Gripper selection results](results/fig_selector.png)

*Left: nothing beats always picking one gripper, and the language model is far worse. Middle: 20% of
objects hold 98% of the success that is available to win. Right: the model ranks objects correctly
(r = +0.41) but almost never predicts the wide gripper.*

### Language models do not help

| model | what it saw | result |
|---|---|---|
| `gpt-oss-120b` (Groq) | object name | 3× worse than always picking one gripper; picks the wide gripper 70% of the time when it is right 27% |
| `llama-3.2-11b-vision` (NVIDIA) | photo of the grasp | answers "2-finger" for all 10 of our objects; says our gripper has 3 fingers |

Better models were not reachable: `nvidia/cosmos-reason2-8b` — NVIDIA's physical-reasoning model and
the natural choice — returns *not found for account*, and `llama-3.2-90b-vision` times out.

**Fine-tuning is not an option** at our scale: LoRA needs roughly 5,000–50,000 examples; we have tens.

## The proxy is the problem

A narrow and a wide parallel jaw differ only in opening width, so one dominates and the choice is
nearly always the same. Our two grippers differ in **contact count and contact area**, which should
be far less lopsided — but that is an expectation until we have our own trials.

## Our own hardware data

**33 grasp videos** extracted into key frames and a motion profile:

| | |
|---|---|
| `data/gripper` | 10 clips, named by object |
| `~/Documents/SP Dutt Video` | 23 grasp clips (of 40 videos; the rest is CAD and lab footage) |
| in air / underwater | 25 / 8 |

`results/grasp_clips.csv` has one row per clip. Configurations and outcomes are **read off the frames
by eye and marked provisional** — 9 high confidence, 9 medium, 15 low. Check them before use.

**Three objects were tried in both configurations**: banana, marker, mug. That is the paired
comparison this step needs. The next trials worth running are the 2-finger versions of egg, pinkball,
rubik and wheel, which already have 4-finger clips.

## Files

| file | what it does |
|---|---|
| `01_extract_labels.py` | per-object success rates for both grippers, from the GraspGen shards |
| `02_train_selector.py` | trains the selector, reports regret against every baseline |
| `03_plot.py` | the figure |
| `04_llm_selector.py` | asks an LLM, scores it against the same labels |
| `05_extract_videos.py` | turns the grasp videos into key frames and a CSV |
| `vlm_gripper_select.py` | asks a vision model about our own gripper photos |
| `force_model.py` | slip vs damage, contact area, position-controlled closing |

```bash
conda activate graspgenx
python step2_gripper_vlm/01_extract_labels.py      # needs the GraspGen shards
python step2_gripper_vlm/02_train_selector.py
python step2_gripper_vlm/03_plot.py
```

## What is needed to finish

Measure the gripper: finger stiffness (press one onto a kitchen scale), contact area (carbon paper),
opening range, closing overshoot. Those four numbers turn `force_model.py` from placeholders into a
real model.

Then paired trials — the same object in both configurations — recording force and whether it was
held, slipped or damaged.
