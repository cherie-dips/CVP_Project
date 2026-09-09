#!/usr/bin/env python3
"""Step 3b. Train a grasp-quality scorer and compare it with GraspGenX's own.

GraspGenX's discriminator scores grasps well for the Franka (per-object AUC 0.664) but below chance
for the Robotiq (0.445). If a small model trained on the same labels beats that, we have a fallback
for when the released model fails on our fin-ray.

Objects are held out, not just grasps — the model is tested on shapes it never saw.

    python step3_grasp_graspgenx/02_train_scorer.py
"""
import json, pathlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
GRASPGENX = {"franka": 0.664, "robotiq": 0.445}     # measured in discriminator_auc.py


def per_object_auc(pred, y, obj):
    """Mean AUC within each object — can it tell good grasps from bad on THIS object?"""
    out = []
    for o in np.unique(obj):
        m = obj == o
        if 0 < y[m].sum() < m.sum():
            out.append(roc_auc_score(y[m], pred[m]))
    return float(np.mean(out)), len(out)


def train(Xtr, ytr, Xva, yva, epochs=80, hidden=128, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    Xtr, ytr = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)[:, None]
    Xva, yva = torch.tensor(Xva), torch.tensor(yva, dtype=torch.float32)[:, None]
    best, state, bad = 1e9, None, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(net(Xtr[b]), ytr[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(nn.functional.binary_cross_entropy_with_logits(net(Xva), yva))
        if v < best - 1e-4:
            best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 10:
                break
    net.load_state_dict(state); net.eval()
    return net


def main():
    d = np.load(RES / "grasp_features.npz", allow_pickle=True)
    X, y, obj, grip = d["X"].astype(np.float32), d["y"].astype(int), d["obj"], d["grip"]
    print(f"{len(X)} grasps, {len(np.unique(obj))} objects, success rate {y.mean():.3f}\n")

    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xn = (X - mu) / sd

    tv, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(X, groups=obj))
    i_tr, i_va = next(GroupShuffleSplit(1, test_size=0.15, random_state=1)
                      .split(tv, groups=obj[tv]))
    tr, va = tv[i_tr], tv[i_va]
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}"
          f"   ({len(np.unique(obj[te]))} held-out objects)\n")

    results = {}
    print(f"{'':<26}{'per-object AUC':>16}{'GraspGenX':>12}")

    # one scorer for both grippers, with the gripper as a feature
    Xg = np.hstack([Xn, (grip == "franka").astype(np.float32)[:, None]])
    net = train(Xg[tr], y[tr], Xg[va], y[va])
    with torch.no_grad():
        pred_all = torch.sigmoid(net(torch.tensor(Xg[te]))).numpy().ravel()

    for g in ("franka", "robotiq"):
        m = grip[te] == g
        auc, n = per_object_auc(pred_all[m], y[te][m], obj[te][m])
        results[g] = dict(ours=auc, graspgenx=GRASPGENX[g], n_objects=n)
        delta = auc - GRASPGENX[g]
        print(f"  {g:<24}{auc:>16.3f}{GRASPGENX[g]:>12.3f}   {delta:+.3f}")

    # what is the signal actually coming from?
    print("\n  feature ablations (Franka, per-object AUC):")
    names = ["position", "approach direction", "surface shape", "jaw occupancy"]
    groups = {"position": list(range(0, 4)), "approach direction": list(range(4, 10)),
              "surface shape": list(range(10, 15)), "jaw occupancy": list(range(15, 18))}
    for nm in names:
        keep = [i for i in range(X.shape[1]) if i not in groups[nm]]
        Xa = np.hstack([Xn[:, keep], (grip == "franka").astype(np.float32)[:, None]])
        na = train(Xa[tr], y[tr], Xa[va], y[va])
        with torch.no_grad():
            pa = torch.sigmoid(na(torch.tensor(Xa[te]))).numpy().ravel()
        mm = grip[te] == "franka"
        auc, _ = per_object_auc(pa[mm], y[te][mm], obj[te][mm])
        print(f"    without {nm:<22}{auc:.3f}   ({auc - results['franka']['ours']:+.3f})")
        results[f"ablate_{nm}"] = auc

    json.dump(results, open(RES / "scorer_metrics.json", "w"), indent=1)
    np.savez_compressed(RES / "scorer_predictions.npz",
                        pred=pred_all, y=y[te], obj=obj[te], grip=grip[te])
    print(f"\nwrote {RES/'scorer_metrics.json'}")


if __name__ == "__main__":
    main()
