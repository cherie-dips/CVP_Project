#!/usr/bin/env python3
"""Figures for Step 2: can we predict which gripper to use?"""
import json, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

RES = pathlib.Path(__file__).resolve().parent / "results"
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8})


def main():
    df = pd.read_csv(RES / "gripper_labels.csv")
    m = json.load(open(RES / "selector_metrics.json"))["results"]
    d = np.load(RES / "selector_predictions.npz", allow_pickle=True)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.79, bottom=0.165, wspace=0.38)

    # (a) how much success is on the table
    ax = axes[0]
    rows = [("always robotiq", m["always robotiq"]["regret"], "#b0342c"),
            ("random", m["random"]["regret"], "#9e9e9e")]
    for shots, lab in ((0, "LLM, no examples"), (16, "LLM, 16 examples")):
        f = RES / f"llm_selector_{shots}shot.json"
        if f.exists():
            j = json.load(open(f))
            rows.append((lab, j["results"][f"LLM {shots}-shot"]["regret"], "#c9a227"))
    rows += [("learned selector", m["learned selector (threshold 0)"]["regret"], "#1a4f7a"),
             ("always franka", m["always franka"]["regret"], "#5b9bd5"),
             ("oracle", 0.0, "#2e7d32")]
    rows.sort(key=lambda r: -r[1])
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
    for i, (_, v, _) in enumerate(rows):
        ax.text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel("regret  (grasp success given up, lower is better)")
    ax.set_xlim(0, 0.30)
    ax.set_title("(a)  Nothing beats just picking one gripper\n     — and the LLM is far worse",
                 loc="left", fontweight="bold")
    ax.tick_params(axis="y", length=0); ax.grid(axis="x", color="0.93"); ax.set_axisbelow(True)

    # (b) where the recoverable success actually lives
    ax = axes[1]
    lost = np.sort((df.oracle_rate - df.franka_rate).values)[::-1]
    cum = np.cumsum(lost) / lost.sum()
    xs = 100 * np.arange(1, len(cum) + 1) / len(cum)
    ax.plot(xs, 100 * cum, color="#1a4f7a", lw=2)
    for p, c in ((5, "#b0342c"), (20, "#c9a227")):
        i = int(len(cum) * p / 100)
        ax.plot([p, p], [0, 100 * cum[i]], color=c, lw=0.9, ls=(0, (3, 3)))
        ax.plot([0, p], [100 * cum[i]] * 2, color=c, lw=0.9, ls=(0, (3, 3)))
        ax.text(p + 1.5, 100 * cum[i] - 7, f"{p}% of objects\nhold {100*cum[i]:.0f}%",
                fontsize=7.8, color=c)
    ax.set_xlim(0, 100); ax.set_ylim(0, 101)
    ax.set_xlabel("objects, ranked by how much the choice matters (%)")
    ax.set_ylabel("share of recoverable success (%)")
    ax.set_title("(b)  The decision only matters\n     for a small minority", loc="left", fontweight="bold")
    ax.grid(color="0.93"); ax.set_axisbelow(True)

    # (c) the model does learn the ranking
    ax = axes[2]
    ax.scatter(d["gap"], d["pred"], s=6, alpha=0.25, color="#1a4f7a", linewidths=0)
    ax.axhline(0, color="0.6", lw=0.8); ax.axvline(0, color="0.6", lw=0.8)
    r = pearsonr(d["pred"], d["gap"])[0]
    ax.set_xlabel("true gap   (franka success − robotiq success)")
    ax.set_ylabel("predicted gap")
    ax.set_title(f"(c)  It ranks correctly (r = {r:+.2f})\n     but the decision rarely flips",
                 loc="left", fontweight="bold")
    ax.grid(color="0.93"); ax.set_axisbelow(True)

    fig.suptitle("Step 2 on the GraspGen proxy pair — 8,454 objects, held-out categories",
                 x=0.105, ha="left", fontsize=10.5, fontweight="bold", y=0.965)
    out = RES / "fig_selector.png"
    fig.savefig(out, dpi=300, facecolor="white"); fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
