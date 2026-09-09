#!/usr/bin/env python3
"""Step 2a. Pull per-object gripper success rates out of the GraspGen shards.

For every object grasped by BOTH grippers, record how often each one succeeded. That success gap is
what the selector has to predict.

    python step2_gripper_vlm/01_extract_labels.py
"""
import json, pathlib, re
import numpy as np
import pandas as pd
import webdataset as wds

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / "data/graspgen"
GRIPPERS = {"franka": "franka_panda", "robotiq": "robotiq_2f_140"}


def main():
    idx = {g: json.load(open(DATA / f"grasp_data/{d}/uuid_index.json"))
           for g, d in GRIPPERS.items()}
    common = set(idx["franka"]) & set(idx["robotiq"])
    print(f"{len(common)} objects grasped by both grippers")

    rec = {}
    for g, d in GRIPPERS.items():
        shards = sorted({idx[g][u] for u in common})
        for s in shards:
            p = DATA / f"grasp_data/{d}/shard_{s:03d}.tar"
            if not p.exists():
                continue
            for sample in wds.WebDataset(str(p), shardshuffle=False):
                u = sample["__key__"]
                if u not in common:
                    continue
                j = json.loads(sample["grasps.json"])
                m = np.asarray(j["grasps"]["object_in_gripper"], bool)
                r = rec.setdefault(u, {})
                r[f"{g}_rate"] = float(m.mean())
                r[f"{g}_n"] = int(m.size)
                r["category"] = re.sub(r".*/simplified/([^/]+)/.*", r"\1", j["object"]["file"])
                r["scale"] = float(j["object"]["scale"])
            print(f"  {d} shard {s}: {len(rec)} objects so far", flush=True)

    df = pd.DataFrame([dict(uuid=u, **v) for u, v in rec.items()])
    df = df.dropna(subset=["franka_rate", "robotiq_rate"])
    df["gap"] = df.franka_rate - df.robotiq_rate
    df["best"] = np.where(df.gap > 0, "franka", "robotiq")
    df["oracle_rate"] = df[["franka_rate", "robotiq_rate"]].max(axis=1)
    out = HERE / "results/gripper_labels.csv"
    df.to_csv(out, index=False)

    print(f"\n{len(df)} objects with both success rates")
    print(f"  franka mean {df.franka_rate.mean():.3f}   robotiq mean {df.robotiq_rate.mean():.3f}")
    print(f"  correlation {df.franka_rate.corr(df.robotiq_rate):.3f}")
    print(f"  franka is better on {(df.gap > 0).mean():.1%} of objects")
    print(f"  oracle {df.oracle_rate.mean():.3f}  vs best fixed choice "
          f"{max(df.franka_rate.mean(), df.robotiq_rate.mean()):.3f}"
          f"  ->  headroom {100*(df.oracle_rate.mean() - max(df.franka_rate.mean(), df.robotiq_rate.mean())):.1f} points")
    print(f"  categories: {df.category.nunique()}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
