#!/usr/bin/env python3
"""Step 2b. Train a model to pick the better gripper for an object.

What matters is not accuracy but REGRET: how much grasp success we give up by choosing wrong. If
both grippers work on an object, picking the "wrong" one costs nothing.

Object description comes from the category name, embedded with CLIP's text encoder, plus the
object's physical size. A robot gets the category from its camera (Step 1 already runs CLIP on the
image), so this is a stand-in for the visual pathway, not a shortcut around it.

    python step2_gripper_vlm/02_train_selector.py
"""
import json, pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from transformers import CLIPModel, CLIPTokenizer

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
MODEL = "openai/clip-vit-base-patch32"


def text_features(categories):
    """CLIP text embedding of a natural phrase for each category."""
    uniq = sorted(set(categories))
    model = CLIPModel.from_pretrained(MODEL).eval()
    tok = CLIPTokenizer.from_pretrained(MODEL)
    prompts = [f"a photo of a {c.replace('_', ' ')}" for c in uniq]
    embs = []
    for i in range(0, len(prompts), 128):
        t = tok(prompts[i:i + 128], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            f = model.get_text_features(**t)
        embs.append(torch.nn.functional.normalize(f, dim=-1).numpy())
    lookup = dict(zip(uniq, np.concatenate(embs).astype(np.float32)))
    return np.stack([lookup[c] for c in categories])


def train_mlp(Xtr, ytr, Xva, yva, epochs=150, hidden=128, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    Xtr, ytr = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)[:, None]
    Xva, yva = torch.tensor(Xva), torch.tensor(yva, dtype=torch.float32)[:, None]
    best, state, bad = 1e9, None, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            nn.functional.mse_loss(net(Xtr[b]), ytr[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(net(Xva), yva))
        if v < best - 1e-5:
            best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 20:
                break
    net.load_state_dict(state)
    net.eval()
    return net


def regret(choice, df):
    """Success given up by not choosing the better gripper, averaged over objects."""
    got = np.where(choice == "franka", df.franka_rate.values, df.robotiq_rate.values)
    return float((df.oracle_rate.values - got).mean())


def main():
    df = pd.read_csv(RES / "gripper_labels.csv")
    print(f"{len(df)} objects, {df.category.nunique()} categories\n")

    T = text_features(df.category.values)
    # `scale` is a mesh multiplier, not a size: raw Objaverse extents span 0.5 to 44,000 units,
    # so scale alone says nothing about how big the object is. Real dimensions come from the mesh.
    geo_cols = ["long", "mid", "short", "aspect", "flatness", "volume"]
    have_geo = all(c in df.columns for c in geo_cols)
    if have_geo:
        G = df[geo_cols].values.astype(np.float32)
        G = np.log(np.abs(G) + 1e-6)
        G = (G - G.mean(0)) / (G.std(0) + 1e-8)
        X = np.hstack([T, G]).astype(np.float32)
        print("features: CLIP text on the category + 6 measured shape features")
    else:
        X = T.astype(np.float32)
        print("features: CLIP text on the category only (no mesh geometry available)")
    y = df.gap.values.astype(np.float32)          # franka rate minus robotiq rate

    # hold out whole categories: the model must generalise to object kinds it never saw
    tv, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0)
                  .split(X, groups=df.category))
    i_tr, i_va = next(GroupShuffleSplit(1, test_size=0.15, random_state=1)
                      .split(tv, groups=df.category.values[tv]))
    tr, va = tv[i_tr], tv[i_va]
    dte = df.iloc[te]
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}"
          f"   (categories shared with train: "
          f"{len(set(df.category.values[te]) & set(df.category.values[tr]))})\n")

    res = {}
    n = len(dte)
    print(f"{'strategy':<40}{'regret':>9}{'accuracy':>10}{'success':>9}")

    def report(name, choice):
        choice = np.asarray(choice)
        r = regret(choice, dte)
        acc = float((choice == dte.best.values).mean())
        got = np.where(choice == "franka", dte.franka_rate.values, dte.robotiq_rate.values).mean()
        res[name] = dict(regret=r, accuracy=acc, success=float(got))
        print(f"{name:<40}{r:>9.4f}{acc:>10.3f}{got:>9.3f}")

    report("always franka", np.full(n, "franka"))
    report("always robotiq", np.full(n, "robotiq"))
    rng = np.random.default_rng(0)
    report("random", rng.choice(["franka", "robotiq"], n))
    # bigger object -> wider gripper, the obvious heuristic
    if have_geo:
        thr = np.median(df["long"].values[tr])
        report("size rule (big -> robotiq)",
               np.where(dte["long"].values > thr, "robotiq", "franka"))

    net = train_mlp(X[tr], y[tr], X[va], y[va])
    with torch.no_grad():
        pred = net(torch.tensor(X[te])).numpy().ravel()
        pred_va = net(torch.tensor(X[va])).numpy().ravel()
    report("learned selector (threshold 0)", np.where(pred > 0, "franka", "robotiq"))

    # One gripper wins 73% of the time, so a symmetric threshold almost never fires.
    # Tune it on the validation split, where it costs nothing to look.
    dva = df.iloc[va]
    grid = np.quantile(pred_va, np.linspace(0.01, 0.99, 99))
    best_t = min(grid, key=lambda t: regret(np.where(pred_va > t, "franka", "robotiq"), dva))
    report(f"learned selector (tuned threshold)", np.where(pred > best_t, "franka", "robotiq"))
    from scipy.stats import pearsonr
    print(f"\n  predicted gap vs true gap: r = {pearsonr(pred, dte.gap.values)[0]:+.3f}"
          f"   (the ranking is informative even when the decision is not)")

    report("oracle (upper bound)", dte.best.values)

    # Aggregate regret is swamped by the objects where one gripper wins trivially.
    # The deployment question is: when the choice ACTUALLY matters, do we get it right?
    for thr_gap in (0.1, 0.2, 0.3):
        m = np.abs(dte.gap.values) > thr_gap
        if m.sum() < 30:
            continue
        sub = dte[m]
        acc_model = float((np.where(pred[m] > best_t, "franka", "robotiq") == sub.best.values).mean())
        acc_fixed = float((sub.best.values == "franka").mean())
        reg_model = regret(np.where(pred[m] > best_t, "franka", "robotiq"), sub)
        reg_fixed = regret(np.full(m.sum(), "franka"), sub)
        res[f"decisive |gap|>{thr_gap}"] = dict(n=int(m.sum()), accuracy=acc_model,
                                                always_franka_accuracy=acc_fixed,
                                                regret=reg_model, always_franka_regret=reg_fixed)
        print(f"  where |gap| > {thr_gap}  (n={m.sum():>4}):  model {acc_model:.3f} vs "
              f"always-franka {acc_fixed:.3f}   regret {reg_model:.3f} vs {reg_fixed:.3f}")
    print()

    best_fixed = min(res["always franka"]["regret"], res["always robotiq"]["regret"])
    best_learned = min(res["learned selector (threshold 0)"]["regret"],
                       res["learned selector (tuned threshold)"]["regret"])
    lift = best_fixed - best_learned
    print(f"\nlearned selector beats the best fixed choice by {100*lift:+.2f} points of success")
    print(f"of the {100*best_fixed:.2f} points the oracle could recover, it captures "
          f"{100*lift/best_fixed if best_fixed else 0:.0f}%")

    json.dump(dict(n_test=n, results=res, lift_points=100 * lift), open(RES / "selector_metrics.json", "w"), indent=1)
    np.savez_compressed(RES / "selector_predictions.npz", pred=pred,
                        franka=dte.franka_rate.values, robotiq=dte.robotiq_rate.values,
                        category=dte.category.values, gap=dte.gap.values)
    print(f"\nwrote {RES/'selector_metrics.json'}")


if __name__ == "__main__":
    main()
