#!/usr/bin/env python3
"""Step 1d. Metrics and diagnostic figures for the mass predictor.

    python step1_properties_nerf2physics/04_evaluate.py
"""
import json, pathlib, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
sys.path.insert(0, str(HERE))
from common import mnre        # noqa: E402

MAIN = "CLIP image + size"
ORDER = ["physics", "category median", "CLIP image", "CLIP image + size", "product title + size"]
COL = {"physics": "#b0342c", "category median": "#9e9e9e", "CLIP image": "#5b9bd5",
       "CLIP image + size": "#1a4f7a", "product title + size": "#c9a227"}
PUBLISHED = {"NeRF2Physics": 0.552, "2D CNN": 0.362, "image2mass": 0.341, "LLaVA": 0.306}

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8})


def load(split):
    z = np.load(RES / "predictions.npz", allow_pickle=True)
    d = {k.split("::", 1)[1]: z[k] for k in z.files if k.startswith(split + "::")}
    true = d.pop("true").astype(float)
    meta = {k: d.pop(k) for k in ("ptype", "name") if k in d}
    preds = {k.replace("pred::", ""): v.astype(float) for k, v in d.items()}
    return true, preds, meta


def metrics(pred, true):
    ratio = np.maximum(pred / true, true / pred)
    return dict(MnRE=mnre(pred, true),
                ADE_kg=float(np.mean(np.abs(pred - true))),
                median_rel_err=float(np.median(np.abs(pred - true) / true)),
                within_1_5x=float(np.mean(ratio < 1.5)),
                within_2x=float(np.mean(ratio < 2.0)),
                spearman=float(spearmanr(pred, true).statistic))


def table():
    rows = []
    for split in ("random", "held_out_categories"):
        true, preds, _ = load(split)
        for m in ORDER:
            rows.append(dict(split=split, method=m, n=len(true), **metrics(preds[m], true)))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "metrics.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return df


def fig_overview(df):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), gridspec_kw={"width_ratios": [1.55, 1]})
    fig.subplots_adjust(left=0.215, right=0.965, top=0.775, bottom=0.145, wspace=0.40)

    ax = axes[0]
    r = df[df.split == "random"].set_index("method")
    h = df[df.split == "held_out_categories"].set_index("method")
    y = np.arange(len(ORDER))
    ax.barh(y, [r.loc[m, "MnRE"] for m in ORDER], color=[COL[m] for m in ORDER],
            height=0.58, zorder=2)
    for i, m in enumerate(ORDER):
        ax.text(r.loc[m, "MnRE"] - 0.012, i, f"{r.loc[m,'MnRE']:.3f}", va="center", ha="right",
                color="white", fontweight="bold", fontsize=8.2, zorder=4)
    ax.scatter([h.loc[m, "MnRE"] for m in ORDER], y, facecolor="white", edgecolor="0.15",
               s=44, zorder=5, linewidths=1.1)
    for i, m in enumerate(ORDER):
        ax.text(h.loc[m, "MnRE"], i - 0.34, f"{h.loc[m,'MnRE']:.3f}", ha="center", va="bottom",
                fontsize=7.6, color="0.15")
    for name, v in PUBLISHED.items():
        ax.axvline(v, color="0.85", lw=0.8, ls=(0, (3, 3)), zorder=1)

    ax.set_yticks(y); ax.set_yticklabels(ORDER); ax.invert_yaxis()
    ax.set_xlim(0, 0.80)
    ax.set_xlabel("MnRE  (higher is better)   ·   grey lines: published methods on ABO-500")
    ax.set_title("(a)  Which method predicts mass best", loc="left", fontweight="bold", pad=22)
    fig.legend(handles=[Patch(facecolor="#1a4f7a", label="random split"),
                        Line2D([], [], marker="o", ls="none", markerfacecolor="white",
                               markeredgecolor="0.15", markersize=7,
                               label="held-out categories (the honest number)")],
               loc="upper left", bbox_to_anchor=(0.215, 0.995), frameon=False, ncol=2,
               handletextpad=0.5, columnspacing=1.8, fontsize=8.5)
    ax.grid(axis="x", color="0.93", zorder=0); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)

    ax = axes[1]
    lc = pd.DataFrame(json.load(open(RES / "learning_curve.json")))
    ax.plot(lc.n, lc.mnre, "o-", color=COL[MAIN], lw=1.8, ms=5)
    ax.axhline(lc.mnre.max(), color="0.7", lw=0.8, ls=(0, (3, 3)))
    ax.set_xscale("log")
    ax.set_xticks([1000, 3000, 10000, 30000])
    ax.set_xticklabels(["1k", "3k", "10k", "30k"])
    ax.minorticks_off()
    ax.set_xlabel("training objects"); ax.set_ylabel("MnRE")
    ax.set_title("(b)  More data stops helping", loc="left", fontweight="bold", pad=8)
    ax.grid(color="0.93"); ax.set_axisbelow(True)

    out = RES / "fig1_overview.png"
    fig.savefig(out, dpi=300, facecolor="white"); fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}")


def fig_diagnostics():
    true, preds, meta = load("held_out_categories")
    p = preds[MAIN]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.17, wspace=0.30)

    ax = axes[0]
    ax.scatter(true, p, s=5, alpha=0.16, color=COL[MAIN], linewidths=0)
    lim = [max(1e-3, true.min()), true.max()]
    ax.plot(lim, lim, color="0.2", lw=1.1)
    for f, ls in ((2.0, (0, (4, 3))), (0.5, (0, (4, 3)))):
        ax.plot(lim, [lim[0] * f, lim[1] * f], color="0.55", lw=0.8, ls=ls)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("true mass (kg)"); ax.set_ylabel("predicted mass (kg)")
    m = metrics(p, true)
    ax.set_title(f"(a)  Predicted vs true    {m['within_2x']:.0%} within 2×",
                 loc="left", fontweight="bold")
    ax.text(0.03, 0.95, "dashed: ±2×", transform=ax.transAxes, fontsize=7.5, color="0.45", va="top")

    ax = axes[1]
    err = np.log(p / true)
    ax.hist(err, bins=60, color=COL[MAIN], alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="0.2", lw=1.1)
    ax.axvline(np.median(err), color="#b0342c", lw=1.4)
    ax.text(np.median(err), ax.get_ylim()[1] * 0.96, f" bias {np.exp(np.median(err)):.2f}×",
            color="#b0342c", fontsize=8, va="top")
    ax.set_xlim(-3, 3)
    ax.set_xlabel("log( predicted / true )"); ax.set_ylabel("objects")
    ax.set_title("(b)  Error distribution", loc="left", fontweight="bold")

    ax = axes[2]
    q = pd.qcut(true, 8, duplicates="drop")
    g = pd.DataFrame({"t": true, "p": p, "q": q}).groupby("q", observed=True)
    xs = [iv.mid for iv in g.groups.keys()]
    ys = [mnre(v.p.values, v.t.values) for _, v in g]
    ns = [len(v) for _, v in g]
    ax.plot(xs, ys, "o-", color=COL[MAIN], lw=1.8, ms=5)
    ax.set_xscale("log")
    ax.set_xlabel("true mass (kg), binned"); ax.set_ylabel("MnRE")
    ax.set_ylim(0, 1)
    ax.set_title("(c)  Accuracy across the mass range", loc="left", fontweight="bold")
    ax.grid(color="0.93"); ax.set_axisbelow(True)

    fig.suptitle("Where the image model succeeds and fails   ·   held-out categories, "
                 f"{len(true)} objects", x=0.06, ha="left", fontsize=10.5, fontweight="bold", y=0.965)
    out = RES / "fig2_diagnostics.png"
    fig.savefig(out, dpi=300, facecolor="white"); fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}")


def fig_by_category():
    true, preds, meta = load("held_out_categories")
    df = pd.DataFrame({"t": true, "p": preds[MAIN], "c": meta["ptype"]})
    g = df.groupby("c").filter(lambda x: len(x) >= 30).groupby("c")
    s = pd.Series({c: mnre(v.p.values, v.t.values) for c, v in g}).sort_values()
    sel = pd.concat([s.head(8), s.tail(8)])
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    fig.subplots_adjust(left=0.40, right=0.97, top=0.90, bottom=0.11)
    cols = ["#b0342c"] * 8 + ["#2e7d32"] * 8
    ax.barh(np.arange(len(sel)), sel.values, color=cols, height=0.65)
    for i, v in enumerate(sel.values):
        ax.text(v + 0.008, i, f"{v:.2f}", va="center", fontsize=7.8)
    ax.set_yticks(np.arange(len(sel)))
    ax.set_yticklabels([c.replace("_", " ").lower()[:30] for c in sel.index])
    ax.invert_yaxis(); ax.set_xlim(0, 1.0)
    ax.set_xlabel("MnRE"); ax.tick_params(axis="y", length=0)
    ax.set_title("Hardest and easiest object categories\n"
                 "held-out categories, ≥30 objects each", loc="left", fontweight="bold")
    ax.grid(axis="x", color="0.93"); ax.set_axisbelow(True)
    out = RES / "fig3_by_category.png"
    fig.savefig(out, dpi=300, facecolor="white"); fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}")


if __name__ == "__main__":
    df = table()
    fig_overview(df)
    fig_diagnostics()
    fig_by_category()
