# Stage 1 — object → size → mass

Given a photograph, estimate how heavy the object is.

## Pipeline

```
  photo ──► segment ──► which pixels are the object
              │
  depth ──────┴──────► size in metres
              │
              ▼
     CLIP image features (512) + size (5)
              │
              ▼
       small net → mass in kg
```

**Why depth is needed.** A photo has no scale — a toy car and a real car look the same. Segmentation
says *which pixels*; depth says *how big*.

**Training data:** ABO, 32,687 products with a photo, real dimensions and a true weight. Tested on
whole product categories the model never saw.

## Results

**MnRE** — for each object, the smaller of (predicted ÷ true) and (true ÷ predicted), averaged. 1.0 is perfect.

| method | MnRE |
|---|---|
| guess the median for its category | 0.268 |
| image only | 0.509 |
| **image + size** | **0.619** |

![Stage 1 results](results/stage1.png)

**How accurate does the depth camera need to be?** Barely accurate at all:

| error in measured size | MnRE |
|---|---|
| 0% | 0.619 |
| 10% | 0.605 |
| 20% | 0.602 |
| 40% | 0.582 |

Even a 40% size error still beats using no size at all (0.509). A cheap depth sensor is enough.

**Could we skip depth?** Predicting size from the photo alone gives MnRE 0.717, a **26% median error** —
which the table above says would cost only ~0.015 MnRE. So a monocular fallback is viable if depth fails.

## Run it

```bash
conda activate graspgenx
jupyter notebook stage1_mass.ipynb
```

Needs `data/abo` (listings metadata + small images) and the precomputed CLIP embeddings in `results/`.
