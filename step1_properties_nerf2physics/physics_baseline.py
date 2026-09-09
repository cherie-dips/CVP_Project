#!/usr/bin/env python3
"""The physics baseline: mass = density x volume.

Fetches real 3-D meshes from ABO, measures occupied volume by voxelisation, then asks whether the
physics formula does better with a true volume than with a bounding box.

    python step1_properties_nerf2physics/physics_baseline.py --fetch    # download meshes first
    python step1_properties_nerf2physics/physics_baseline.py
"""
import argparse, io, json, pathlib, sys, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
import trimesh
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from common import density_of, mnre        # noqa: E402
BASE = "https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original/"


def fill_fraction(mesh, n_vox=48):
    """Occupied volume / bounding box, scale-free. Voxelise (handles open surfaces), else hull."""
    bb = float(np.prod(mesh.bounding_box.extents))
    if bb <= 0:
        return None, None
    hull = float(mesh.convex_hull.volume) / bb
    try:
        pitch = max(mesh.bounding_box.extents) / n_vox
        v = mesh.voxelized(pitch).fill().volume / bb
        if 0 < v <= 1.0:
            return float(v), hull
    except Exception:
        pass
    return (hull if 0 < hull <= 1.0 else None), hull


def fetch(args):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max_mb", type=float, default=25.0)
    ap.add_argument("--budget_gb", type=float, default=3.0)
    ap.add_argument("--out", default=str(HERE / "results/abo_mesh_volumes.csv"))
    a = args

    models = pd.read_csv(REPO / "data/abo/3dmodels.csv.gz")
    listings = pd.read_csv(HERE / "results/abo_dataset.csv")
    j = models.merge(listings, left_on="3dmodel_id", right_on="item_id")
    j = j[(j.faces > 500) & (j.faces < 200000)]
    j = j.sample(frac=1.0, random_state=0)          # random order, not size-biased
    print(f"{len(j)} candidates with a mesh and a true weight; fetching up to {a.n}\n")

    def fetch(r):
        url = BASE + r["path"]
        try:
            # size check BEFORE downloading: a 60 MB glb otherwise costs a minute to discard
            h = requests.head(url, timeout=30)
            mb = int(h.headers.get("content-length", 0)) / 1e6
            if not (0 < mb <= a.max_mb):
                return None
            content = requests.get(url, timeout=180).content
            mesh = trimesh.load(io.BytesIO(content), file_type="glb", force="mesh")
            fill, hull = fill_fraction(mesh)
            if fill is None:
                return None
            return dict(item_id=r["3dmodel_id"], name=r["name"], material=r["material"],
                        product_type=r["product_type"], mass_kg=r["mass_kg"],
                        bbox_vol=r["bbox_vol"], fill=fill, hull_fill=hull,
                        true_vol=fill * r["bbox_vol"], faces=int(r["faces"]), mb=mb)
        except Exception:
            return None

    rows, used = [], 0.0
    cands = [r for _, r in j.head(a.n * 6).iterrows()]
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(fetch, r): r for r in cands}
        for fut in as_completed(futs):
            got = fut.result()
            if got is None:
                continue
            rows.append(got); used += got["mb"] * 1e6
            if len(rows) % 25 == 0:
                print(f"  {len(rows):>4} meshes, {used/1e9:.2f} GB, "
                      f"median fill {np.median([x['fill'] for x in rows]):.3f}", flush=True)
            if len(rows) >= a.n or used / 1e9 > a.budget_gb:
                break
        for f in futs:
            f.cancel()

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"\n{len(df)} meshes, {used/1e9:.2f} GB downloaded")
    print(f"fill fraction: median {df.fill.median():.3f}, "
          f"5th-95th {df.fill.quantile(.05):.3f}-{df.fill.quantile(.95):.3f}")
    print(f"wrote {a.out}")




def fit_and_score(tr, te, vol_col, dens_col):
    """One global calibration constant fitted on train, then MnRE on test."""
    k = float(np.median(tr.mass_kg / (tr[dens_col] * tr[vol_col])))
    pred = k * te[dens_col].values * te[vol_col].values
    return mnre(pred, te.mass_kg.values), k


def evaluate():
    df = pd.read_csv(HERE / "results/abo_mesh_volumes.csv")
    df["density"] = df.material.map(density_of)
    df["const_density"] = 800.0
    print(f"{len(df)} objects with a real mesh and a true weight")
    print(f"  with a usable material label: {df.density.notna().sum()}")
    print(f"  fill fraction (true volume / bbox): median {df.fill.median():.3f}, "
          f"5th-95th {df.fill.quantile(.05):.3f}-{df.fill.quantile(.95):.3f}\n")

    tr, te = train_test_split(df, test_size=0.3, random_state=0)
    y = te.mass_kg.values
    res = {}

    print(f"{'method':<46}{'MnRE':>8}")
    res["global median"] = mnre(np.full(len(te), tr.mass_kg.median()), y)
    print(f"{'guess the median mass':<46}{res['global median']:>8.3f}")
    med = tr.groupby("product_type").mass_kg.median()
    res["category median"] = mnre(te.product_type.map(med).fillna(tr.mass_kg.median()).values, y)
    print(f"{'median mass for its category':<46}{res['category median']:>8.3f}")

    print(f"\n{'  --- physics: mass = k x density x volume ---':<46}")
    for label, vol in (("bounding box volume", "bbox_vol"), ("TRUE mesh volume", "true_vol")):
        s, k = fit_and_score(tr, te, vol, "const_density")
        res[f"physics, {label}, constant density"] = s
        print(f"{'  ' + label + ', constant density':<46}{s:>8.3f}   (k={k:.3f})")

    m = df.dropna(subset=["density"])
    if len(m) > 40:
        mtr, mte = train_test_split(m, test_size=0.3, random_state=0)
        print(f"\n{'  --- with a real material label (n=' + str(len(m)) + ') ---':<46}")
        for label, vol in (("bounding box volume", "bbox_vol"), ("TRUE mesh volume", "true_vol")):
            k = float(np.median(mtr.mass_kg / (mtr.density * mtr[vol])))
            s = mnre(k * mte.density.values * mte[vol].values, mte.mass_kg.values)
            res[f"physics+material, {label}"] = s
            print(f"{'  ' + label + ', material density':<46}{s:>8.3f}   (k={k:.3f})")

    gain = (res["physics, TRUE mesh volume, constant density"]
            - res["physics, bounding box volume, constant density"])
    print(f"\nreal volume changes the physics route by {gain:+.3f} MnRE")
    json.dump(dict(n=len(df), results=res, gain_from_true_volume=gain),
              open(HERE / "results/true_volume_test.json", "w"), indent=1)
    print(f"wrote {HERE/'results/true_volume_test.json'}")




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download meshes first (~1 GB)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max_mb", type=float, default=12.0)
    ap.add_argument("--budget_gb", type=float, default=3.0)
    ap.add_argument("--out", default=str(HERE / "results/abo_mesh_volumes.csv"))
    a = ap.parse_args()
    if a.fetch:
        fetch(a)
    evaluate()
