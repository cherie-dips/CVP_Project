#!/usr/bin/env python3
"""Step 1c. Train the mass predictor and save test-set predictions.

A robot's camera gives a photograph, not an Amazon product title, so the model that matters maps
IMAGE -> MASS. The product title is trained too, but only as an upper bound we cannot deploy.

    python step1_properties_nerf2physics/03_train.py
"""
import json, pathlib, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit, train_test_split

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import density_of, mnre        # noqa: E402


def train_mlp(Xtr, ytr, Xva, yva, epochs=120, hidden=256, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(hidden, hidden // 2), nn.ReLU(),
                        nn.Linear(hidden // 2, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)[:, None]
    Xva_t, yva_t = torch.tensor(Xva), torch.tensor(yva, dtype=torch.float32)[:, None]
    best, best_state, patience = 1e9, None, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            nn.functional.smooth_l1_loss(net(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(nn.functional.smooth_l1_loss(net(Xva_t), yva_t))
        if v < best - 1e-4:
            best, best_state, patience = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 15:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def predict(net, X):
    with torch.no_grad():
        return net(torch.tensor(X)).numpy().ravel()


def main():
    d = np.load(HERE / "results/abo_clip_embeddings.npz", allow_pickle=True)
    X_img = d["X"].astype(np.float32)
    mass = d["mass_kg"].astype(np.float64)
    vol = d["bbox_vol"].astype(np.float64)
    dims = np.stack([d["h"], d["w"], d["l"]], 1).astype(np.float32)
    names, ptype = d["name"].astype(str), d["product_type"].astype(str)
    material = d["material"].astype(str)
    y = np.log(mass)
    n = len(mass)
    print(f"{n} objects with a photograph and a true weight\n")

    geo = np.column_stack([np.log(vol), np.log(dims + 1e-6),
                           dims.max(1) / (dims.min(1) + 1e-6)]).astype(np.float32)
    geo = (geo - geo.mean(0)) / (geo.std(0) + 1e-8)
    Xig = np.hstack([X_img, geo])

    # two splits: a plain random one, and one that holds out whole product categories.
    # Catalogues contain near-duplicate variants (the same chair in five colours), so a random
    # split leaks them across train and test. The category split is the honest number.
    def random_split():
        tv, te = train_test_split(np.arange(n), test_size=0.2, random_state=0)
        tr, va = train_test_split(tv, test_size=0.15, random_state=0)
        return tr, va, te

    def category_split():
        tv, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0)
                      .split(np.arange(n), groups=ptype))
        i_tr, i_va = next(GroupShuffleSplit(1, test_size=0.15, random_state=1)
                          .split(tv, groups=ptype[tv]))
        return tv[i_tr], tv[i_va], te

    out = {}
    for split_name, split_fn in (("random", random_split),
                                 ("held_out_categories", category_split)):
        tr, va, te = (np.asarray(x) for x in split_fn())
        shared = len(set(ptype[te]) & set(ptype[tr]))
        print(f"--- {split_name}:  train {len(tr)}  val {len(va)}  test {len(te)}"
              f"   (categories shared with train: {shared})")
        p = {}
        med = pd.Series(mass[tr]).groupby(pd.Series(ptype[tr])).median()
        p["category median"] = pd.Series(ptype[te]).map(med).fillna(np.median(mass[tr])).values
        dens = np.array([density_of(m) if m != "nan" else np.nan for m in material], float)
        k = np.nanmedian(mass[tr] / (np.nan_to_num(dens[tr], nan=800.0) * vol[tr]))
        p["physics"] = k * np.nan_to_num(dens[te], nan=800.0) * vol[te]
        p["CLIP image"] = np.exp(predict(train_mlp(X_img[tr], y[tr], X_img[va], y[va]), X_img[te]))
        p["CLIP image + size"] = np.exp(predict(train_mlp(Xig[tr], y[tr], Xig[va], y[va]), Xig[te]))
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        Ttr, Tte = vec.fit_transform(names[tr]), vec.transform(names[te])
        r = Ridge(1.0).fit(hstack([Ttr, csr_matrix(geo[tr])]), y[tr])
        p["product title + size"] = np.exp(r.predict(hstack([Tte, csr_matrix(geo[te])])))
        for kname, pred in p.items():
            print(f"    {kname:<26} MnRE {mnre(pred, mass[te]):.3f}")
        out[split_name] = dict(true=mass[te], ptype=ptype[te], name=names[te],
                               **{f"pred::{a}": b for a, b in p.items()})

    np.savez_compressed(HERE / "results/predictions.npz",
                        **{f"{s}::{k}": v for s, d_ in out.items() for k, v in d_.items()})
    print(f"\nwrote {HERE/'results/predictions.npz'}")

    # learning curve on the honest split
    print("\nlearning curve (held-out categories, CLIP image + size)")
    tr_i, va_i, g_te = (np.asarray(x) for x in category_split())
    curve = []
    rng = np.random.default_rng(0)
    for frac in (0.05, 0.1, 0.25, 0.5, 1.0):
        sub = rng.choice(tr_i, max(50, int(len(tr_i) * frac)), replace=False)
        m = train_mlp(Xig[sub], y[sub], Xig[va_i], y[va_i])
        s = mnre(np.exp(predict(m, Xig[g_te])), mass[g_te])
        curve.append(dict(n=int(len(sub)), mnre=float(s)))
        print(f"    n={len(sub):>6}  MnRE {s:.3f}")
    json.dump(curve, open(HERE / "results/learning_curve.json", "w"), indent=1)
    print(f"wrote {HERE/'results/learning_curve.json'}")


if __name__ == "__main__":
    main()
