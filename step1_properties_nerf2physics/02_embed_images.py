#!/usr/bin/env python3
"""CLIP image embeddings for ABO objects that have a ground-truth weight.

This is the deployable setting: a camera gives an image, not an Amazon product title.

    python step1_properties_nerf2physics/02_embed_images.py --limit 0
"""
import argparse, pathlib, tarfile, warnings
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
IMG_ROOT = REPO / "data/abo/images/small"
MODEL = "openai/clip-vit-base-patch32"


def pick_device(model, proc):
    """MPS has given wrong numbers elsewhere in this repo, so verify before trusting it."""
    if not torch.backends.mps.is_available():
        return "cpu"
    x = Image.new("RGB", (224, 224), (120, 90, 60))
    inp = proc(images=[x], return_tensors="pt")
    with torch.no_grad():
        a = model.to("cpu").get_image_features(**inp)[0]
        b = model.to("mps").get_image_features(**{k: v.to("mps") for k, v in inp.items()})[0].cpu()
    d = float((a - b).abs().max())
    print(f"  MPS vs CPU max abs difference: {d:.2e}  -> {'using mps' if d < 1e-2 else 'FALLING BACK to cpu'}")
    return "mps" if d < 1e-2 else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default=str(HERE / "results/abo_clip_embeddings.npz"))
    a = ap.parse_args()

    df = pd.read_csv(HERE / "results/abo_dataset.csv")
    if a.limit:
        df = df.head(a.limit)
    print(f"{len(df)} objects to embed")

    model = CLIPModel.from_pretrained(MODEL).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)
    dev = pick_device(model, proc)
    model = model.to(dev)

    embs, keep = [], []
    batch_imgs, batch_idx = [], []

    def flush():
        if not batch_imgs:
            return
        inp = proc(images=batch_imgs, return_tensors="pt")
        with torch.no_grad():
            f = model.get_image_features(**{k: v.to(dev) for k, v in inp.items()})
        f = torch.nn.functional.normalize(f, dim=-1).cpu().numpy().astype(np.float32)
        embs.append(f); keep.extend(batch_idx)
        batch_imgs.clear(); batch_idx.clear()

    for i, r in enumerate(df.itertuples()):
        p = IMG_ROOT / r.path
        if not p.exists():
            continue
        try:
            batch_imgs.append(Image.open(p).convert("RGB")); batch_idx.append(r.Index)
        except Exception:
            continue
        if len(batch_imgs) >= a.batch:
            flush()
            if len(keep) % 3200 == 0:
                print(f"  {len(keep)}/{len(df)}", flush=True)
    flush()

    X = np.concatenate(embs) if embs else np.zeros((0, 512), np.float32)
    sub = df.loc[keep]
    np.savez_compressed(a.out, X=X, item_id=sub.item_id.values,
                        mass_kg=sub.mass_kg.values, bbox_vol=sub.bbox_vol.values,
                        h=sub.h.values, w=sub.w.values, l=sub.l.values,
                        name=sub.name.values.astype(str),
                        product_type=sub.product_type.values.astype(str),
                        material=sub.material.astype(str).values)
    print(f"\nembedded {len(X)} images -> {a.out}")


if __name__ == "__main__":
    main()
