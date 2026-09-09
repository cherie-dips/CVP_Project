#!/usr/bin/env python3
"""Figure: GraspGenX discriminator score vs. GraspGen ground-truth success labels.

    python step3_grasp_graspgenx/plot_auc_results.py
"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = pathlib.Path(__file__).resolve().parent / "results"
GRIPPERS = [("franka", "Franka Panda", "#2b6ca3", "o"),
            ("robotiq", "Robotiq 2F-140", "#b0342c", "s")]

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
})


def load(name):
    d = json.load(open(RES / f"auc_{name}.json"))
    a = np.array([o["auc"] for o in d["objects"]], float)
    p = np.array([o["pos_rate"] for o in d["objects"]], float)
    ok = ~np.isnan(a)
    return a[ok], p[ok]


def main():
    data = {k: load(k) for k, *_ in GRIPPERS}
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    fig.subplots_adjust(left=0.125, right=0.975, top=0.93, bottom=0.155)

    bins = np.arange(0.10, 1.001, 0.05)
    for k, lab, col, _ in GRIPPERS:
        a, _ = data[k]
        ax.hist(a, bins=bins, color=col, alpha=0.45, edgecolor=col, linewidth=0.9,
                label=f"{lab}   {a.mean():.3f} ± {a.std(ddof=1):.3f}  (n = {a.size})")
    for k, _, col, _ in GRIPPERS:
        ax.axvline(data[k][0].mean(), color=col, lw=1.5)

    ax.axvline(0.5, color="0.20", lw=1.1, ls=(0, (4, 3)), zorder=4)
    ax.text(0.492, ax.get_ylim()[1] * 0.40, "chance", rotation=90, ha="right", va="center",
            fontsize=8, color="0.20")

    ax.set_xlabel("AUC per object")
    ax.set_ylabel("Number of objects")
    ax.set_xlim(0.10, 1.0)
    ax.set_xticks(np.arange(0.1, 1.01, 0.1))
    ax.legend(loc="upper left", frameon=False, handlelength=1.1, handletextpad=0.5,
              borderpad=0.2, labelspacing=0.35)

    out = RES / "auc_results.png"
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    print(f"wrote {out} and {out.with_suffix('.pdf').name}")


if __name__ == "__main__":
    main()
