"""Can the camera replace the force sensor? Predict grip force from the shape of the
marker chain on the finger, scored on objects the model has not seen.

Input per frame (extract.py): the 6 marker positions, and the SEN0704 force in N.
Shape features are measured against the finger's resting shape in the same clip
(the median shape while the sensor reads 0, the way a scale is tared), after
removing translation, rotation and scale, because the gripper is hand-held and
moves around the image.

The camera image trails the sensor GUI by about LAG frames (0.2 s), so the shape
at frame t is paired with the force at frame t - LAG.

    python stage3_force/scripts/force_model.py
        -> results/force_model_metrics.csv, results/force_from_markers.png
"""
import os, sys, warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from crop import RESULTS
from extract import CONTACT_N, N_MARKERS, OUT as FRAMES

LAG = 6
# Accelerate's BLAS on macOS raises spurious overflow warnings inside sklearn's matmul;
# the ridge fit matches the closed-form solution to 1e-15.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
FEATURES = ["deformation", "deformation_max", "bend_net", "bend_total", "bend_ends", "bend_middle",
            "sagitta", "chord"]


# ── shape features ──────────────────────────────────────────────────────────
def _normalise(P):
    """Centre each 6-marker shape and scale it to unit length along the chain."""
    P = P - P.mean(1, keepdims=True)
    return P / np.linalg.norm(np.diff(P, axis=1), axis=2).sum(1)[:, None, None]


def _procrustes_residual(P, R):
    """P aligned to R by the best rotation and scale, minus R."""
    U, S, Vt = np.linalg.svd(P.T @ R)
    D = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
    s = np.trace(np.diag(S) @ D) / (P ** 2).sum()
    return s * P @ (U @ D @ Vt) - R


def _turning(P):
    """Turning angle at each of the 4 inner markers."""
    a = np.arctan2(*np.diff(P, axis=-2)[..., ::-1].swapaxes(0, -1))
    return (np.diff(a.T, axis=-1) + np.pi) % (2 * np.pi) - np.pi


def _sagitta(P):
    """Signed distance of each marker from the line through the end markers, over its length."""
    chord = P[..., -1, :] - P[..., 0, :]
    L = np.linalg.norm(chord, axis=-1, keepdims=True)
    normal = np.stack([-chord[..., 1], chord[..., 0]], -1) / L
    return np.einsum("...ij,...j->...i", P - P[..., :1, :], normal) / L


def features(df):
    rows = []
    for k, d in df[df.markers_found].groupby("clip"):
        P = _normalise(np.stack([d[[f"m{j}x", f"m{j}y"]].values for j in range(N_MARKERS)], 1))
        rest = d.force_now.values == 0
        ref = P[np.argmax(rest)] if rest.any() else P[0]
        flip = np.linalg.norm(P[:, ::-1] - ref, axis=(1, 2)) < np.linalg.norm(P - ref, axis=(1, 2))
        P[flip] = P[flip, ::-1]                               # same end first in every frame
        R = _normalise(np.median(P[rest] if rest.sum() >= 10 else P, 0)[None])[0]
        res = np.stack([_procrustes_residual(p, R) for p in P])
        th, th0 = _turning(P), _turning(R)
        sg, sg0 = _sagitta(P), _sagitta(R)
        side = np.sign(sg0[np.argmax(np.abs(sg0))]) or 1.0    # which way the finger bows at rest
        rows.append(pd.DataFrame({
            "clip": k, "object": d.object.values, "frame": d.frame.values, "t": d.t.values,
            "force_N": d.force_N.values,
            "deformation": np.sqrt((res ** 2).sum((1, 2))),           # Procrustes distance to rest
            "deformation_max": np.linalg.norm(res, axis=2).max(1),    # largest single-marker shift
            "bend_net": np.abs(th.sum(1)) - abs(th0.sum()),
            "bend_total": np.abs(th).sum(1) - np.abs(th0).sum(),
            "bend_ends": np.abs(th[:, [0, 3]]).sum(1) - np.abs(th0[[0, 3]]).sum(),
            "bend_middle": np.abs(th[:, [1, 2]]).sum(1) - np.abs(th0[[1, 2]]).sum(),
            "sagitta": (side * sg[:, 1:-1]).max(1) - np.abs(sg0).max(),
            "chord": np.linalg.norm(P[:, -1] - P[:, 0], axis=1) - np.linalg.norm(R[-1] - R[0]),
        }))
    return pd.concat(rows, ignore_index=True)


def load():
    df = pd.read_csv(FRAMES)
    df["force_now"] = df.force_N
    # pair the shape at t with the force LAG frames earlier
    df["force_N"] = df.groupby("clip").force_N.shift(LAG)
    df = df[df.force_N.notna()]
    return features(df)


# ── evaluation ──────────────────────────────────────────────────────────────
MODELS = {
    "mean of training force": None,
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "gradient boosting": lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05,
                                                               min_samples_leaf=40, random_state=0),
}


def leave_one_object_out(F):
    preds = {name: pd.Series(np.nan, index=F.index) for name in MODELS}
    for obj in F.object.unique():
        test = F.object == obj
        X, y = F.loc[~test, FEATURES].values, F.loc[~test, "force_N"].values
        for name, make in MODELS.items():
            if make is None:
                preds[name][test] = y.mean()
            else:
                preds[name][test] = np.clip(make().fit(X, y).predict(F.loc[test, FEATURES].values), 0, None)
    return preds


def score(F, preds):
    rows = []
    for (k, obj), d in F.groupby(["clip", "object"]):
        y = d.force_N.values
        contact = y > CONTACT_N
        a, b = np.polyfit(d.deformation, y, 1) if y.std() > 0 else (0, 0)
        fit = a * d.deformation + b
        row = {"clip": k, "object": obj, "frames": len(d), "contact_frames": int(contact.sum()),
               "peak_N": round(y.max(), 2),
               # how well force follows deformation when fitted to this clip alone (in-sample)
               "within_clip_r2": round(1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum(), 3)
               if y.std() > 0 else np.nan}
        for name, p in preds.items():
            p = p[d.index].values
            tag = name.split()[0]
            row[f"{tag}_mae_N"] = round(np.abs(p - y).mean(), 3)
            if name != "mean of training force":
                row[f"{tag}_spearman"] = round(spearmanr(p, y)[0], 3) if y.std() > 0 else np.nan
                row[f"{tag}_contact_auc"] = round(roc_auc_score(contact, p), 3) \
                    if 0 < contact.sum() < len(y) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot(F, preds, metrics, out, model="gradient boosting"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frames = pd.read_csv(FRAMES)
    clips = sorted(F["clip"].unique())
    fig, axes = plt.subplots(len(clips), 1, figsize=(12, 1.9 * len(clips)), sharex=False)
    for ax, k in zip(axes, clips):
        d = frames[frames["clip"] == k]
        f = F[F["clip"] == k]
        ax.plot(d.t, d.force_N.shift(LAG), color="0.25", lw=1, label="sensor (calibrated)")
        ax.plot(f.t, preds[model][f.index], ".", ms=2, color="C3", label=f"camera only ({model}, object held out)")
        ax2 = ax.twinx()
        ax2.plot(f.t, 100 * f.deformation, ".", ms=1.2, color="C0", alpha=.5)
        ax2.set_ylabel("deform. %", color="C0", fontsize=8); ax2.tick_params(labelsize=7, colors="C0")
        m = metrics[metrics["clip"] == k].iloc[0]
        ax.set_title(f"clip {k}: {m.object}    within-clip R² {m.within_clip_r2:.2f}    "
                     f"held-out MAE {m.gradient_mae_N:.2f} N (baseline {m.mean_mae_N:.2f} N)",
                     loc="left", fontsize=9)
        ax.set_ylabel("N", fontsize=8); ax.tick_params(labelsize=7)
        ax.set_xlim(d.t.min(), d.t.max())
    axes[0].legend(loc="upper left", fontsize=7, markerscale=4)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main():
    F = load()
    preds = leave_one_object_out(F)
    metrics = score(F, preds)
    metrics.to_csv(os.path.join(RESULTS, "force_model_metrics.csv"), index=False)
    plot(F, preds, metrics, os.path.join(RESULTS, "force_from_markers.png"))
    contact = F.force_N > CONTACT_N
    print(f"{len(F)} frames with all 6 markers, {contact.sum()} of them in contact\n")
    print("held out by object, all frames pooled:")
    for name, p in preds.items():
        line = f"  {name:24s} MAE {np.abs(p - F.force_N).mean():.2f} N   in contact {np.abs(p - F.force_N)[contact].mean():.2f} N"
        if name != "mean of training force":
            line += f"   contact AUC {roc_auc_score(contact, p):.2f}"
        print(line)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\n" + metrics.to_string(index=False))


if __name__ == "__main__":
    main()
