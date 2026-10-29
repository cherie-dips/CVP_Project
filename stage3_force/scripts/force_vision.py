"""Grip force from the camera image of the finger, with a pretrained vision model.

force_model.py predicted force from the six markers alone and did not beat predicting the mean on
objects it had not seen. This uses the whole finger instead: the crops from finger_crops.py show the
contact beam, the ribs, the back beam and the clamp, so the model can see the beam bend and also see
it pushed back relative to the rigid gripper.

The image model is DINOv2 (ViT-B/14 with registers), frozen. Self-supervised features like these
are the strongest choice at small label counts (Sparsh, CoRL 2024: DINOv2-style features beat
end-to-end training when labels are scarce), and an end-to-end CNN that learns objects rather than
deformation is exactly what generalised poorly to unseen objects in the closest prior work (Zuo et
al., RA-L 2026, ResNet-50 + transformer baseline). On top of the features:

    mean        the training mean (the baseline to beat)
    ridge       ridge regression on a PCA of one frame's features
    temporal    a small causal temporal CNN over the last second of features (the "memory" that
                Forces for Free, Sci. Robotics 2025, adds to handle friction and occlusion)

Every model is scored leaving one object out. As in force_model.py the camera lags the sensor by
LAG frames, so the image at frame t is paired with the force at t - LAG.

    python stage3_force/scripts/force_vision.py features   # ~5 min on the M4 GPU, cached
    python stage3_force/scripts/force_vision.py heads      # -> results/force_vision_metrics.csv
                                                           #    results/force_vision.png
"""
import os, sys, time, warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from crop import RESULTS
from extract import CONTACT_N
from finger_crops import VISION
from force_model import LAG

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
BACKBONE = "vit_base_patch14_reg4_dinov2.lvd142m"
FEATS = os.path.join(VISION, "feats_dinov2_b14.npy")
MAX_GAP = 15                 # frames: use a crop only this close to a frame where the markers were found
PCS = 256
ALPHAS = [1e1, 1e2, 1e3, 1e4, 1e5]
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


# ── frozen features ─────────────────────────────────────────────────────────
def features(batch=48):
    """Per frame: class token, mean patch token, and the 16 x 16 patch tokens pooled to 4 x 4."""
    import timm
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model(BACKBONE, pretrained=True, img_size=224, num_classes=0).eval().to(dev)
    crops = np.load(os.path.join(VISION, "crops.npy"), mmap_mode="r")
    n, p = len(crops), model.num_prefix_tokens
    out = np.lib.format.open_memmap(FEATS, mode="w+", dtype=np.float16, shape=(n, 18 * model.embed_dim))
    t = time.time()
    with torch.inference_mode():
        for i in range(0, n, batch):
            x = (np.asarray(crops[i:i + batch], np.float32) / 255 - MEAN) / STD
            tok = model.forward_features(torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)).float().cpu()
            cls, patches = tok[:, 0], tok[:, p:]
            grid = patches.reshape(len(x), 4, 4, 4, 4, -1).mean((2, 4)).flatten(1)     # (B, 16 C)
            out[i:i + len(x)] = torch.cat([cls, patches.mean(1), grid], 1).numpy()
            if i % (batch * 40) == 0:
                print(f"  {i + len(x)}/{n}  {(i + len(x)) / (time.time() - t):.0f} img/s", flush=True)
    out.flush()
    print(f"features {out.shape} -> {FEATS}")


# ── data ────────────────────────────────────────────────────────────────────
def load():
    idx = pd.read_csv(os.path.join(VISION, "index.csv"))
    idx["y"] = idx.groupby("clip").force_N.shift(LAG)            # the force the image at t shows
    idx["use"] = idx.y.notna() & (idx.gap <= MAX_GAP)
    return idx, np.load(FEATS, mmap_mode="r")


class TemporalCNN(torch.nn.Module):
    """Causal dilated 1-D CNN over per-frame features: receptive field 29 frames (~1 s)."""

    def __init__(self, d_in, width=128, p=0.2):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i, dil in enumerate((1, 2, 4)):
            self.layers.append(torch.nn.Conv1d(d_in if i == 0 else width, width, 5, dilation=dil))
        self.drop, self.head = torch.nn.Dropout(p), torch.nn.Conv1d(width, 1, 1)

    def forward(self, x):                      # x: (1, d_in, T)
        for conv in self.layers:
            x = self.drop(torch.nn.functional.gelu(conv(torch.nn.functional.pad(x, ((conv.kernel_size[0] - 1)
                                                                                   * conv.dilation[0], 0)))))
        return self.head(x)[0, 0]


def fit_temporal(seqs, d_in, epochs=150, seed=0):
    torch.manual_seed(seed)
    net = TemporalCNN(d_in)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(epochs):
        net.train()
        for j in np.random.default_rng(seed + _).permutation(len(seqs)):
            X, y, m = seqs[j]
            x = X + 0.1 * torch.randn_like(X)
            loss = torch.nn.functional.huber_loss(net(x.T[None])[m], y[m], delta=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


def leave_one_object_out(idx, F):
    preds = {name: np.full(len(idx), np.nan) for name in ("mean", "ridge", "temporal")}
    for obj in idx.object.unique():
        t0 = time.time()
        test, train = idx.object.values == obj, (idx.object.values != obj)
        tr = train & idx.use.values
        scaler = StandardScaler().fit(F[tr].astype(np.float32))
        pca = PCA(PCS, svd_solver="randomized", random_state=0).fit(scaler.transform(F[tr].astype(np.float32)))
        Z = pca.transform(scaler.transform(np.asarray(F, np.float32)))
        Z /= Z[tr].std(0)
        y = idx.y.values

        preds["mean"][test] = y[tr].mean()

        # ridge: alpha by leave-one-object-out inside the training objects
        objs = idx.object.values
        best = min(ALPHAS, key=lambda a: np.mean([
            np.abs(Ridge(alpha=a).fit(Z[tr & (objs != o)], y[tr & (objs != o)]).predict(Z[tr & (objs == o)])
                   - y[tr & (objs == o)]).mean() for o in np.unique(objs[tr])]))
        preds["ridge"][test] = np.clip(Ridge(alpha=best).fit(Z[tr], y[tr]).predict(Z[test]), 0, None)

        # temporal CNN over whole clips (frames with an unusable crop are kept as input, not as targets)
        seqs = []
        for k in np.unique(idx["clip"].values[train]):
            c = idx["clip"].values == k
            seqs.append((torch.from_numpy(Z[c].astype(np.float32)), torch.from_numpy(np.nan_to_num(y[c]).astype(np.float32)),
                         torch.from_numpy(idx.use.values[c])))
        net = fit_temporal(seqs, PCS)
        with torch.no_grad():
            for k in np.unique(idx["clip"].values[test]):
                c = idx["clip"].values == k
                preds["temporal"][c] = np.clip(net(torch.from_numpy(Z[c].astype(np.float32)).T[None]).numpy(), 0, None)
        print(f"  held out {obj:16s} ridge alpha {best:g}   {time.time() - t0:.0f} s", flush=True)
    return preds


# ── scoring ─────────────────────────────────────────────────────────────────
def scores(idx, preds, mask):
    rows = []
    for name, p in preds.items():
        y, q = idx.y.values[mask], p[mask]
        contact = y > CONTACT_N
        rows.append({"model": name, "frames": int(mask.sum()), "mae_N": round(np.abs(q - y).mean(), 3),
                     "mae_contact_N": round(np.abs(q - y)[contact].mean(), 3),
                     "spearman": round(spearmanr(q, y)[0], 3) if name != "mean" else np.nan,
                     "contact_auc": round(roc_auc_score(contact, q), 3) if name != "mean" else np.nan})
    return pd.DataFrame(rows)


NAMES = {"ridge": "DINOv2 features + linear regression", "temporal": "DINOv2 features + temporal CNN"}


def plot(idx, preds, out, model="ridge"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clips = sorted(idx["clip"].unique())
    fig, axes = plt.subplots(len(clips), 1, figsize=(12, 1.9 * len(clips)))
    for ax, k in zip(axes, clips):
        d = idx[idx["clip"] == k]
        u = d.use.values
        ax.plot(d.t, d.y, color="0.25", lw=1, label="force sensor (measured)")
        ax.plot(d.t[u], preds[model][d.index][u], ".", ms=2, color="C3",
                label=f"predicted from the camera image ({NAMES[model]}), object not seen in training")
        mae = np.abs(preds[model][d.index][u] - d.y[u]).mean()
        base = np.abs(preds["mean"][d.index][u] - d.y[u]).mean()
        ax.set_title(f"clip {k}: {d.object.iloc[0]}    average error {mae:.2f} N    "
                     f"(always predicting the training average: {base:.2f} N)", loc="left", fontsize=9)
        ax.set_ylabel("N", fontsize=8); ax.tick_params(labelsize=7); ax.set_xlim(d.t.min(), d.t.max())
    axes[0].legend(loc="upper left", fontsize=7, markerscale=4)
    fig.suptitle("Grip force from the camera image of the finger, on objects the model never saw",
                 fontsize=12, fontweight="bold")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=110)


def heads():
    idx, F = load()
    preds = leave_one_object_out(idx, F)
    pd.DataFrame({"clip": idx["clip"], "frame": idx.frame, "y": idx.y, "use": idx.use, **preds}).to_csv(
        os.path.join(VISION, "preds_frozen.csv"), index=False)
    usable = idx.use.values
    marked = usable & idx.markers_found.values.astype(bool)
    table = pd.concat([scores(idx, preds, usable).assign(frames_set="usable crops"),
                       scores(idx, preds, marked).assign(frames_set="all 6 markers (as force_model.py)")])
    table.to_csv(os.path.join(RESULTS, "force_vision_metrics.csv"), index=False)
    plot(idx, preds, os.path.join(RESULTS, "force_vision.png"))
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False))


if __name__ == "__main__":
    {"features": features, "heads": heads}[sys.argv[1]]()
