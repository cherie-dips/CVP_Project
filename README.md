# Picking a Gripper from What an Object Is Made Of

Given a photograph of an object, decide **which gripper to use, where to grip it, and how hard**.

Two fin-ray grippers: a **2-finger** and a **4-finger**. They are the same gripper — the 2-finger has
two fingers removed — so material, finger length and stiffness are identical and only the number of
contacts changes.

## Pipeline

```
  photo + depth
       │
       ▼
  STAGE 1   object → size → mass                      MnRE 0.619
       │
       ▼
  STAGE 2   object + mass → which gripper             regret 0.0500
       │
       ▼
  STAGE 3   object + mass + gripper → grasp points    AUC 0.607
```

## Results

| stage | result |
|---|---|
| **1 — mass from a photo** | **MnRE 0.619** with size, 0.509 without. A 40% error in the measured size still beats using no size at all, so a cheap depth sensor suffices. |
| **2 — which gripper** | **regret 0.0500**, which only matches always picking one gripper (0.0505). Adding mass changes nothing. |
| **3 — where to grip** | Proposals land **2–9 mm** from the object surface. Scorer reaches **AUC 0.607**. The 4-finger is correctly found infeasible on objects too big for its 148 mm envelope. |

![Stage 1](stage1_mass/results/stage1.png)
![Stage 2](stage2_gripper/results/stage2.png)

Each stage has its own README with the method and the full numbers.

## The honest limitation

**Mass does not help Stages 2 or 3 — but that cannot be tested on public data.** Every public grasping
dataset uses *one fixed density and friction for all objects*. GraspGen and MultiGripperGrasp both do.
So the labels contain no mass information, and the core claim of this pipeline is untestable on them
by construction.

**Only our own trials can settle it.** We have 33 grasp clips and **3 objects tried in both gripper
configurations**. More paired trials is the single most valuable thing to collect.

## Layout

```
stage1_mass/       stage1_mass.ipynb       photo + depth → mass
stage2_gripper/    stage2_gripper.ipynb    → which gripper
stage3_grasp/      stage3_grasp.ipynb      → grasp points
data/              abo/ graspgen/ gripper/ grasp_clips/
gripper/           our gripper's STL files
papers/            background reading
```

## Environment

```bash
conda activate graspgenx
jupyter notebook
```

Do not install OpenCV — it pulls numpy 2.x and breaks the pinned 1.26.4. The video tools use ffmpeg
and PIL instead.

## Data

| | size | used by |
|---|---|---|
| `data/abo` | 3.6 GB | Stage 1 — 32,687 products with photos and true weights |
| `data/graspgen` | 14 GB | Stages 2 and 3 — 8,454 objects with labelled grasps |
| `data/gripper`, `data/grasp_clips` | 7 MB | our 33 grasp videos and their key frames |

Both large sets are raw downloads; everything extracted from them is saved in `stage*/results/`, so
they can be deleted and re-fetched if space is needed.
