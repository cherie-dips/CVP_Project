#!/usr/bin/env python3
"""Step 1a. Build the training table: name, material, size, true weight, and a photo path.

    python step1_properties_nerf2physics/01_build_dataset.py
"""
import gzip, json, glob, pathlib, re
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "data/abo/listings/metadata"
OUT = REPO / "step1_properties_nerf2physics/results/abo_dataset.csv"

TO_KG = {"kilograms": 1.0, "grams": 1e-3, "milligrams": 1e-6,
         "pounds": 0.45359237, "ounces": 0.028349523}
TO_M = {"meters": 1.0, "centimeters": 1e-2, "millimeters": 1e-3,
        "micrometer": 1e-6, "inches": 0.0254, "feet": 0.3048}


def _val(field, table):
    """Pull a normalized (value, unit) pair and convert to SI."""
    if not field:
        return None
    v = field.get("normalized_value") or field
    unit, value = v.get("unit"), v.get("value")
    if unit not in table or value is None:
        return None
    return float(value) * table[unit]


def _text(field):
    if not field:
        return None
    for e in field:
        if e.get("language_tag", "en").startswith("en"):
            return e.get("value")
    return field[0].get("value")


def main():
    rows = []
    for f in sorted(glob.glob(str(SRC / "*.json.gz"))):
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            m = _val((r.get("item_weight") or [None])[0], TO_KG)
            D = r.get("item_dimensions") or {}
            dims = [_val(D.get(k), TO_M) for k in ("height", "width", "length")]
            if m is None or any(d is None for d in dims):
                continue
            if not (1e-4 < m < 500) or any(not (1e-3 < d < 5) for d in dims):
                continue                      # drop impossible listings
            name = _text(r.get("item_name")) or ""
            mat = _text(r.get("material"))
            ptype = (r.get("product_type") or [{}])[0].get("value")
            rows.append(dict(
                item_id=r.get("item_id"), name=name[:200],
                material=mat.strip().lower() if mat else None,
                product_type=ptype,
                h=dims[0], w=dims[1], l=dims[2],
                bbox_vol=dims[0] * dims[1] * dims[2],
                mass_kg=m,
            ))
    df = pd.DataFrame(rows).dropna(subset=["mass_kg", "bbox_vol"])
    df["apparent_density"] = df.mass_kg / df.bbox_vol

    # attach the product photograph
    main = {}
    for f in sorted(glob.glob(str(SRC / "*.json.gz"))):
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            if r.get("main_image_id"):
                main[r["item_id"]] = r["main_image_id"]
    imgs = pd.read_csv(REPO / "data/abo/images.csv.gz")
    df["main_image_id"] = df.item_id.map(main)
    df = df.merge(imgs[["image_id", "path"]], left_on="main_image_id",
                  right_on="image_id", how="inner")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"rows (with mass, size and a photo)   {len(df)}")
    print(f"with material       {df.material.notna().sum()}")
    print(f"product types       {df.product_type.nunique()}")
    print(f"mass kg   median {df.mass_kg.median():.3f}   range {df.mass_kg.min():.4f}-{df.mass_kg.max():.1f}")
    print(f"bbox vol  median {df.bbox_vol.median():.5f} m3")
    print(f"apparent density  median {df.apparent_density.median():.0f} kg/m3"
          f"  (water = 1000; lower means the box is mostly air)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
