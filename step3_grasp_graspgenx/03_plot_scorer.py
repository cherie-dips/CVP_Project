#!/usr/bin/env python3
"""Figure: our grasp scorer versus GraspGenX's."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score

RES = pathlib.Path(__file__).resolve().parent / "results"
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8})
OURS, THEIRS = "#1a4f7a", "#b0342c"


def main():
    m = json.load(open(RES / "scorer_metrics.json"))
    d = np.load(RES / "scorer_predictions.npz", allow_pickle=True)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.78, bottom=0.16, wspace=0.30)

    # (a) head to head
    ax = axes[0]
    g = ["franka", "robotiq"]
    x = np.arange(2); w = 0.36
    ax.bar(x - w/2, [m[k]["graspgenx"] for k in g], w, color=THEIRS, label="GraspGenX (released)")
    ax.bar(x + w/2, [m[k]["ours"] for k in g], w, color=OURS, label="ours (18 features)")
    for i, k in enumerate(g):
        ax.text(i - w/2, m[k]["graspgenx"] + .012, f"{m[k]['graspgenx']:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, m[k]["ours"] + .012, f"{m[k]['ours']:.3f}", ha="center", fontsize=8)
    ax.axhline(0.5, color="0.3", lw=1.1, ls=(0, (4, 3)))
    ax.text(-0.45, 0.515, "chance", ha="left", fontsize=7.5, color="0.3")
    ax.set_xticks(x); ax.set_xticklabels(["Franka Panda\n(80 mm)", "Robotiq 2F-140\n(140 mm)"])
    ax.set_ylim(0, 0.78); ax.set_ylabel("per-object AUC")
    ax.set_title("(a)  Ours recovers the gripper\n     GraspGenX fails on", loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", color="0.93"); ax.set_axisbelow(True)

    # (b) which features carry the signal
    ax = axes[1]
    base = m["franka"]["ours"]
    keys = [k for k in m if k.startswith("ablate_")]
    drops = sorted(((k.replace("ablate_", ""), base - m[k]) for k in keys), key=lambda t: t[1])
    y = np.arange(len(drops))
    ax.barh(y, [v for _, v in drops], color=[OURS if v > 0 else "0.75" for _, v in drops], height=0.6)
    for i, (_, v) in enumerate(drops):
        ax.text(v + 0.0015 * np.sign(v or 1), i, f"{v:+.3f}", va="center", fontsize=8)
    ax.axvline(0, color="0.3", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels([k for k, _ in drops])
    ax.set_xlabel("AUC lost when the feature is removed")
    ax.set_title("(b)  Local surface shape carries\n     most of the signal", loc="left", fontweight="bold")
    ax.tick_params(axis="y", length=0); ax.grid(axis="x", color="0.93"); ax.set_axisbelow(True)

    # (c) spread across objects
    ax = axes[2]
    for name, col in (("franka", "#5b9bd5"), ("robotiq", THEIRS)):
        sel = d["grip"] == name
        aucs = []
        for o in np.unique(d["obj"][sel]):
            mm = sel & (d["obj"] == o)
            if 0 < d["y"][mm].sum() < mm.sum():
                aucs.append(roc_auc_score(d["y"][mm], d["pred"][mm]))
        ax.hist(aucs, bins=np.arange(0.1, 1.01, 0.06), alpha=0.55, color=col, label=name,
                edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color="0.3", lw=1.1, ls=(0, (4, 3)))
    ax.set_xlabel("per-object AUC"); ax.set_ylabel("objects")
    ax.set_title("(c)  It works on most objects,\n     but not all", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")

    fig.suptitle("Step 3 — scoring grasps: 89,512 labelled grasps, 57 held-out objects",
                 x=0.06, ha="left", fontsize=10.5, fontweight="bold", y=0.965)
    out = RES / "fig_scorer.png"
    fig.savefig(out, dpi=300, facecolor="white"); fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
