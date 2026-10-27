"""Label, render and split every object the gripper selector is trained and tested on.

The first fine-tune had 100 objects, 26 of them for training and 3 four-finger objects in the
test set, so its 0% recall was 0 out of 3. This rebuilds the dataset at a size where per-class
numbers mean something:

    label   label_geometry.label, unchanged: 4-finger when the cross-section across the long axis
            has circularity >= 0.70, dropped when even the short side is wider than the gripper
    images  views.axis_views (down the long axis, then the two side views) and views.fixed_views
            (two oblique views, the layout the first fine-tune used), grey, written to
            data/selector/{axis,fixed}/<id>.png
    splits  the Objaverse objects 60/15/25 train/valid/test, stratified by label, seed 0; the 50
            real-scanned objects (26 YCB, 24 Google Scanned Objects) are a second test set,
            never trained on

    /opt/anaconda3/envs/graspgenx/bin/python stage1_gripper/scripts/build_selector_dataset.py
        -> stage1_gripper/results/selector/objects.csv
"""
import json, multiprocessing as mp, pathlib, sys, traceback, warnings

import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"stage1_gripper"/"scripts"), str(ROOT/"stage2_grasp"/"scripts")]
from label_geometry import features, label
from views import axis_views, fixed_views
import objects

IMAGES = ROOT/"data"/"selector"
OUT = ROOT/"stage1_gripper"/"results"/"selector"
SPLIT = {"train": 0.60, "valid": 0.15, "test": 0.25}


def catalogue():
    ann = json.loads((ROOT/"data"/"objaverse"/"annotations.json").read_text())
    scales = json.loads((ROOT/"data"/"objaverse"/"scales.json").read_text())
    rows = [dict(id=p.stem, source="objaverse", name=ann.get(p.stem, {}).get("name", ""), up="y")
            for p in sorted((ROOT/"data"/"objaverse"/"meshes").glob("*.glb")) if p.stem in scales]
    rows += [dict(id=f"ycb_{p.name}", source="ycb", name=p.name.split("_", 1)[1].replace("_", " "), up="z")
             for p in sorted((ROOT/"data"/"ycb").iterdir()) if (p/"google_16k"/"nontextured.ply").exists()]
    rows += [dict(id=f"gso_{p.name}", source="gso", name=p.name.replace("_", " "), up="z")
             for p in sorted((ROOT/"selected_objects").iterdir())
             if (p/f"{p.name}.usd").exists() and not p.name[:3].isdigit()]      # YCB copies skipped
    return rows


def load(row):
    if row["source"] == "objaverse":
        return objects.objaverse(row["id"])
    if row["source"] == "ycb":
        return objects.ycb(row["id"][4:])
    name = row["id"][4:]
    return objects.usd(ROOT/"selected_objects"/name/f"{name}.usd")


def process(row):
    warnings.filterwarnings("ignore")
    try:
        mesh = load(row)
        if len(mesh.faces) == 0 or mesh.area <= 0:
            return {**row, "error": "empty mesh"}
        f = features(mesh, seed=0)
        gripper, _ = label(f)
        out = {**row, **f, "faces": len(mesh.faces), "label": gripper}
        if gripper is None:                                 # wider than the gripper opens
            return out
        axis, fixed = IMAGES/"axis"/f"{row['id']}.png", IMAGES/"fixed"/f"{row['id']}.png"
        if not axis.exists():
            axis_views(mesh).save(axis)
        if not fixed.exists():
            fixed_views(mesh, up=(0, 1, 0) if row["up"] == "y" else (0, 0, 1)).save(fixed)
        return out
    except Exception as e:
        return {**row, "error": f"{type(e).__name__}: {e}"[:200], "trace": traceback.format_exc()[-400:]}


def split(df):
    """Stratified by label, so every split keeps the natural 4-finger rate."""
    rng = np.random.default_rng(0)
    df["split"] = "external"
    obj = df.source == "objaverse"
    for lab in ["2-finger", "4-finger"]:
        idx = rng.permutation(df.index[obj & (df.label == lab)].to_numpy())
        a = int(round(SPLIT["train"] * len(idx)))
        b = a + int(round(SPLIT["valid"] * len(idx)))
        df.loc[idx[:a], "split"], df.loc[idx[a:b], "split"], df.loc[idx[b:], "split"] = "train", "valid", "test"
    return df


def main():
    for d in ("axis", "fixed"):
        (IMAGES/d).mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = catalogue()
    print(f"{len(rows)} objects: " + ", ".join(f"{s} {sum(r['source'] == s for r in rows)}"
                                             for s in ("objaverse", "ycb", "gso")))
    with mp.get_context("spawn").Pool(8) as pool:
        done = []
        for i, r in enumerate(pool.imap_unordered(process, rows, chunksize=4), 1):
            done.append(r)
            if i % 200 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
    df = pd.DataFrame(done).sort_values(["source", "id"]).reset_index(drop=True)
    bad = df["error"].notna() if "error" in df else pd.Series(False, index=df.index)
    for _, r in df[bad].iterrows():
        print("  skipped", r["id"], r["error"])
    df = df[~bad & df["label"].notna()].drop(columns=[c for c in ("error", "trace") if c in df]).reset_index(drop=True)
    df = split(df)
    df.to_csv(OUT/"objects.csv", index=False)
    print(f"\n{bad.sum()} failed to load, {len(done) - bad.sum() - len(df)} wider than the gripper")
    print(pd.crosstab(df.split, df.label, margins=True).to_string())


if __name__ == "__main__":
    main()
