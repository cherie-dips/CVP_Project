"""Fine-tune an ImageNet ResNet-18 end to end on the finger crops, scored leaving one object out.

The end-to-end alternative to force_vision.py's frozen DINOv2 features, the approach of camera-based
force estimators such as VFTS (Collins et al., IROS 2023) and the baseline in Zuo et al. (RA-L 2026).
Augmentation is strong on colour (so the network cannot simply recognise the pink or orange cube) and
mild on geometry (the crop already follows the finger). Same labels as force_vision.py: the image at
frame t against the force at t - LAG, crops within MAX_GAP frames of a marker frame.

Resumable: each held-out object's predictions are saved as soon as its fold finishes.

    python stage3_force/scripts/force_finetune.py
        -> data/forceSensor/vision/preds_finetune.csv
           results/force_finetune_log.csv    loss per epoch per fold
"""
import math, os, sys, time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
import torchvision

sys.path.insert(0, os.path.dirname(__file__))
from crop import RESULTS
from finger_crops import VISION
from force_vision import MEAN, STD, load, scores

EPOCHS, BATCH, LR, STRIDE = 8, 64, 3e-4, 2      # STRIDE: train on every 2nd frame; 30 fps neighbours repeat
PREDS = os.path.join(VISION, "preds_finetune.csv")
LOG = os.path.join(RESULTS, "force_finetune_log.csv")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def augment(x, g):
    """x: (B, 3, H, W) in [0, 1] on the device. Colour jitter, small rotation/shift/zoom, random erasing."""
    B = len(x)
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(B, 1, 1, 1, device=x.device, generator=g)
    grey = x.mean(1, keepdim=True)
    x = (x - grey) * r(0.6, 1.4) + grey                                   # saturation
    x = (x - x.mean((1, 2, 3), keepdim=True)) * r(0.7, 1.3) + x.mean((1, 2, 3), keepdim=True)   # contrast
    x = x * r(0.7, 1.3)                                                   # brightness
    x = x[:, torch.randperm(3, generator=g, device=x.device)] if torch.rand(1, generator=g, device=x.device) < .3 else x
    a = (r(-8, 8).flatten() * math.pi / 180)
    s = r(0.92, 1.08).flatten()
    t = r(-0.04, 0.04).reshape(B, 1) * torch.ones(B, 2, device=x.device)
    theta = torch.stack([torch.stack([torch.cos(a) / s, -torch.sin(a) / s, t[:, 0]], 1),
                         torch.stack([torch.sin(a) / s, torch.cos(a) / s, t[:, 1]], 1)], 1)
    x = Fn.grid_sample(x, Fn.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
    erase = torch.rand(B, device=x.device, generator=g) < .25                # a random 20-40 % patch
    for i in torch.where(erase)[0].tolist():
        h, w = (int(x.shape[2] * (.2 + .2 * torch.rand(1, generator=g, device=x.device).item())) for _ in range(2))
        y0, x0 = (int(torch.randint(0, x.shape[2] - h, (1,), generator=g, device=x.device)),
                  int(torch.randint(0, x.shape[3] - w, (1,), generator=g, device=x.device)))
        x[i, :, y0:y0 + h, x0:x0 + w] = torch.rand(3, 1, 1, device=x.device, generator=g)
    return x.clamp(0, 1)


def batches(crops, ids, y=None, train=False, g=None):
    mean = torch.tensor(MEAN, device=DEV)[:, None, None]
    std = torch.tensor(STD, device=DEV)[:, None, None]
    order = ids[torch.randperm(len(ids)).numpy()] if train else ids
    for i in range(0, len(order), BATCH):
        b = np.sort(order[i:i + BATCH])
        x = torch.from_numpy(np.asarray(crops[b])).to(DEV).permute(0, 3, 1, 2).float() / 255
        if train:
            x = augment(x, g)
        yield b, (x - mean) / std, (torch.from_numpy(y[b]).float().to(DEV) if y is not None else None)


def model():
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Linear(net.fc.in_features, 1)
    return net.to(DEV)


def fold(idx, crops, obj, log):
    y, objs, use = idx.y.values.astype(np.float32), idx.object.values, idx.use.values
    train = np.where((objs != obj) & use)[0][::STRIDE]
    test = np.where(objs == obj)[0]
    torch.manual_seed(0)
    g = torch.Generator(device=DEV).manual_seed(0)
    net = model()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * math.ceil(len(train) / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.15)
    for ep in range(EPOCHS):
        net.train(); t, tot, n = time.time(), 0.0, 0
        for _, x, yb in batches(crops, train, y, train=True, g=g):
            loss = Fn.huber_loss(net(x)[:, 0], yb, delta=1.0)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot, n = tot + loss.item() * len(yb), n + len(yb)
        log.append({"held_out": obj, "epoch": ep + 1, "train_huber": round(tot / n, 4), "seconds": round(time.time() - t, 1)})
        pd.DataFrame(log).to_csv(LOG, index=False)
        print(f"  {obj:16s} epoch {ep + 1}/{EPOCHS}  huber {tot / n:.3f}  {time.time() - t:.0f} s", flush=True)
    net.eval(); out = np.empty(len(test), np.float32)
    with torch.no_grad():
        for b, x, _ in batches(crops, test):
            out[np.searchsorted(test, b)] = net(x)[:, 0].clamp(min=0).cpu().numpy()
    return test, out


def main():
    idx, _ = load()
    crops = np.load(os.path.join(VISION, "crops.npy"), mmap_mode="r")
    pred = pd.read_csv(PREDS).finetune.values if os.path.exists(PREDS) else np.full(len(idx), np.nan)
    log = pd.read_csv(LOG).to_dict("records") if os.path.exists(LOG) else []
    for obj in idx.object.unique():
        if not np.isnan(pred[idx.object.values == obj]).all():
            continue                                                       # done in an earlier run
        test, out = fold(idx, crops, obj, log)
        pred[test] = out
        pd.DataFrame({"clip": idx["clip"], "frame": idx.frame, "y": idx.y, "use": idx.use, "finetune": pred}).to_csv(PREDS, index=False)
    frozen = pd.read_csv(os.path.join(VISION, "preds_frozen.csv"))
    preds = {"mean": frozen["mean"].values, "ridge (frozen DINOv2)": frozen.ridge.values, "fine-tuned ResNet-18": pred}
    usable = idx.use.values
    marked = usable & idx.markers_found.values.astype(bool)
    table = pd.concat([scores(idx, preds, usable).assign(frames_set="usable crops"),
                       scores(idx, preds, marked).assign(frames_set="all 6 markers (as force_model.py)")])
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
