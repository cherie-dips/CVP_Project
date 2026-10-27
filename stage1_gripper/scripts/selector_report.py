"""Score every selector run on the same held-out objects, with per-class numbers and their uncertainty.

Reads results/selector/objects.csv and the scores_<tag>.csv files vlm_selector.py writes. For each
run, the decision threshold is either 0 (what greedy decoding does) or the one that maximises
balanced accuracy on the validation split; test and external numbers never touch that choice.

Also fits a linear probe (balanced logistic regression) on the frozen VLM's hidden state at the
answer position, to separate "the information is not in the image features" from "the model does
not read it out".

    /opt/anaconda3/envs/graspgenx/bin/python stage1_gripper/scripts/selector_report.py
        -> results/selector/metrics.csv, results/selector/selector_results.png
"""
import pathlib, warnings

import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT/"stage1_gripper"/"results"/"selector"
FEATURES = ROOT/"data"/"selector"
warnings.filterwarnings("ignore", category=RuntimeWarning)       # Accelerate BLAS false alarms

RUNS = {   # tag -> what the run is
    "fixed_zeroshot": "Qwen3-VL-2B, zero-shot, two oblique views",
    "axis_zeroshot": "Qwen3-VL-2B, zero-shot, views along the principal axes",
    "fixed_lora": "Qwen3-VL-2B, LoRA, two oblique views",
    "axis_lora": "Qwen3-VL-2B, LoRA, views along the principal axes",
    "smol_axis_zeroshot": "SmolVLM2-500M, zero-shot, views along the principal axes",
    "smol_axis_lora": "SmolVLM2-500M, LoRA, views along the principal axes",
}


def best_threshold(y, s):
    cands = np.unique(s)
    cands = np.concatenate([[cands[0] - 1], (cands[1:] + cands[:-1]) / 2, [cands[-1] + 1]])
    ba = [((s[y == 1] > t).mean() + (s[y == 0] <= t).mean()) / 2 for t in cands]
    return float(cands[int(np.argmax(ba))])


def metrics(y, s, thr):
    p = (s > thr).astype(int)
    r4, r2 = p[y == 1].mean(), 1 - p[y == 0].mean()
    prec = p[(p == 1)].size and (y[p == 1] == 1).mean()
    return dict(n=len(y), n_4finger=int(y.sum()), accuracy=(p == y).mean(), balanced_accuracy=(r4 + r2) / 2,
                recall_2finger=r2, recall_4finger=r4, precision_4finger=float(prec),
                mcc=matthews_corrcoef(y, p) if 0 < p.sum() < len(p) else 0.0,
                auc=roc_auc_score(y, s), says_4finger=p.mean())


def bootstrap(y, s, thr, n=2000, seed=0):
    """95% intervals, resampling objects within each class so both classes keep their size."""
    rng = np.random.default_rng(seed)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    out = {k: [] for k in ("balanced_accuracy", "recall_4finger", "recall_2finger", "auc")}
    for _ in range(n):
        idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        m = metrics(y[idx], s[idx], thr)
        for k in out:
            out[k].append(m[k])
    return {f"{k}_ci": (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def probe_scores(tag, objects):
    """Balanced logistic regression on frozen hidden states; returns a score for every object."""
    f = FEATURES/f"features_{tag}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    X = pd.DataFrame(z["X"], index=z["ids"]).loc[objects.id].values
    y = (objects.label == "4-finger").astype(int).values
    tr = (objects.split == "train").values
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=np.logspace(-4, 1, 12), cv=5, class_weight="balanced", scoring="roc_auc", max_iter=5000))
    clf.fit(X[tr], y[tr])
    return pd.DataFrame({"id": objects.id, "split": objects.split, "label": objects.label,
                         "score": clf.decision_function(X)})


def main():
    objects = pd.read_csv(OUT/"objects.csv")
    runs = {}
    for tag in RUNS:
        f = OUT/f"scores_{tag}.csv"
        if f.exists():
            runs[tag] = pd.read_csv(f)
    for tag in [t for t in runs if t.endswith("zeroshot")]:
        p = probe_scores(tag, objects)
        if p is not None:
            runs[f"{tag}_probe"] = p
            RUNS[f"{tag}_probe"] = RUNS[tag].replace("zero-shot", "frozen, linear probe")

    rows = []
    for tag, df in runs.items():
        df = df.set_index("id")
        df = df.loc[[i for i in objects.id if i in df.index]].reset_index()     # LoRA runs skip train
        y = (df.label == "4-finger").astype(int).values
        s = df.score.values
        va = (df.split == "valid").values
        for rule, thr in (("threshold 0", 0.0), ("validation threshold", best_threshold(y[va], s[va]))):
            if tag.endswith("_probe") and rule == "threshold 0":
                continue                                         # a probe's 0 is arbitrary
            for split in ("test", "external"):
                m = (df.split == split).values
                rows.append(dict(run=tag, description=RUNS[tag], decision=rule, threshold=thr, split=split,
                                 **metrics(y[m], s[m], thr), **bootstrap(y[m], s[m], thr)))
    res = pd.DataFrame(rows)
    res.to_csv(OUT/"metrics.csv", index=False, float_format="%.4f")
    show = res[["run", "decision", "split", "n", "n_4finger", "accuracy", "balanced_accuracy",
                "recall_2finger", "recall_4finger", "mcc", "auc", "says_4finger"]]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(show.to_string(index=False))
    plot(res, objects)


def plot(res, objects):
    """One row per selector: its recall on each class, so a constant answer shows as 100% / 0%."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [(t, d) for t, d in [
        ("fixed_zeroshot", "threshold 0"), ("fixed_zeroshot", "validation threshold"),
        ("axis_zeroshot", "threshold 0"), ("axis_zeroshot", "validation threshold"),
        ("axis_zeroshot_probe", "validation threshold"),
        ("fixed_lora", "validation threshold"), ("axis_lora", "validation threshold"),
        ("smol_axis_zeroshot", "threshold 0"), ("smol_axis_zeroshot", "validation threshold"),
        ("smol_axis_zeroshot_probe", "validation threshold"), ("smol_axis_lora", "validation threshold"),
    ] if not res[(res.run == t) & (res.decision == d)].empty]
    names = []
    for t, d in rows:
        n = RUNS[t].replace("Qwen3-VL-2B, ", "Qwen3-VL-2B  ").replace("SmolVLM2-500M, ", "SmolVLM2-500M  ")
        names.append(n + ("" if t.endswith("probe") else ("  ·  greedy" if d == "threshold 0" else "  ·  calibrated")))
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.48 * len(rows) + 1.5), sharey=True)
    for ax, split, title in ((axes[0], "test", "Objaverse test split"),
                             (axes[1], "external", "YCB + Google Scanned, never trained on")):
        sub = res[res.split == split]
        for i, (t, d) in enumerate(rows):
            r = sub[(sub.run == t) & (sub.decision == d)].iloc[0]
            for k, (key, col) in enumerate((("recall_4finger", "#1f6fd0"), ("recall_2finger", "#e8a33d"))):
                y = i + (k - 0.5) * 0.36
                ax.barh(y, r[key], height=0.34, color=col)
                lo, hi = r[f"{key}_ci"] if isinstance(r[f"{key}_ci"], tuple) else eval(r[f"{key}_ci"])
                ax.plot([lo, hi], [y, y], color="k", lw=0.8)
            ax.text(1.03, i, f"bal. acc. {r.balanced_accuracy:.0%}\nAUC {r.auc:.2f}", va="center", fontsize=7.5)
        ax.set_xlim(0, 1.25); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
        ax.axvline(1, color=".85", lw=.8)
        n, n4 = int(sub.n.iloc[0]), int(sub.n_4finger.iloc[0])
        ax.set_title(f"{title}\n{n} objects, {n4} need 4 fingers", fontsize=10, loc="left")
        ax.set_xlabel("recall, with 95% bootstrap interval")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#1f6fd0", "#e8a33d")]
    fig.legend(handles, ["4-finger objects found", "2-finger objects found"], loc="lower right",
               fontsize=8, frameon=False, ncol=2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT/"selector_results.png", dpi=150, facecolor="white")
    print("saved", (OUT/"selector_results.png").relative_to(ROOT))


def gallery(run="axis_lora", n=6, seed=1):
    """Test objects as the selector sees them, with the rule's label and the selector's pick."""
    from PIL import Image, ImageDraw
    objects = pd.read_csv(OUT/"objects.csv")
    s = pd.read_csv(OUT/f"scores_{run}.csv").set_index("id")
    va = s[s.split == "valid"]
    thr = best_threshold((va.label == "4-finger").astype(int).values, va.score.values)
    t = objects[objects.split.isin(["test", "external"])].sample(frac=1, random_state=seed)
    pick = pd.concat([t[t.label == "4-finger"].head(n), t[t.label == "2-finger"].head(n)])
    tiles = []
    for r in pick.itertuples():
        im = Image.open(FEATURES/"axis"/f"{r.id}.png").convert("RGB")
        im = im.resize((im.width // 2, im.height // 2))
        canvas = Image.new("RGB", (im.width, im.height + 30), "white")
        canvas.paste(im, (0, 30))
        guess = "4-finger" if s.score[r.id] > thr else "2-finger"
        d = ImageDraw.Draw(canvas)
        d.text((4, 2), f"{(r.name if isinstance(r.name, str) and r.name else r.id)[:44]}", fill=(0, 0, 0))
        d.text((4, 15), f"rule {r.label} (circ. {r.circularity:.2f})  ·  selector {guess}",
               fill=(0, 120, 0) if guess == r.label else (190, 30, 30))
        tiles.append(canvas)
    W, H = tiles[0].size
    sheet = Image.new("RGB", (2 * W + 10, n * H), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i // n) * (W + 10), (i % n) * H))
    sheet.save(OUT/"examples.png")
    print("saved", (OUT/"examples.png").relative_to(ROOT))


if __name__ == "__main__":
    main()
