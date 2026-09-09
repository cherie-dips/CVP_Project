#!/usr/bin/env python3
"""Does GraspGenX's geometric score predict real grasp success?

Scores the GraspGen dataset's labelled grasps with GraspGenX's discriminator and computes AUC
against the Isaac-Sim `object_in_gripper` labels. This decides whether s_geo is worth using as a
feature in PhysHead:

    AUC > 0.75    s_geo is a good feature — build on it as planned
    AUC 0.6-0.75  usable but weak — lean harder on property features
    AUC ~ 0.5     uninformative — drop it, train PhysHead on the point cloud directly

    conda activate graspgenx
    python stage2_discriminator_auc.py --gripper franka_panda --shard 6 --n_objects 50
"""
import argparse, json, logging, os, pathlib, re

os.environ.setdefault("GRASPGENX_DEVICE", "cpu")   # MPS is numerically wrong on this machine

REPO = pathlib.Path(__file__).resolve().parents[1]      # repo root, whatever the cwd
import numpy as np
import torch
import trimesh
import webdataset as wds
from sklearn.metrics import roc_auc_score

from graspgenx import get_checkpoints_version_dir
from graspgenx.grasp_server import GraspGenXSampler, _DEVICE
from graspgenx.dataset.dataset import collate
from graspgenx.utils.checkpoint_io import load_model_cfg

# GraspGen dataset gripper name -> the matching GraspGenX gripper config
GRIPPER_MAP = {"franka_panda": "franka_panda", "robotiq_2f_140": "robotiq_2f_140"}


def iter_objects(shard_path, uuid_filter):
    """Yield every matching object; the caller stops once it has SCORED enough."""
    for s in wds.WebDataset(shard_path, shardshuffle=False):
        k = s["__key__"]
        if uuid_filter and k not in uuid_filter:
            continue
        yield k, json.loads(s["grasps.json"])


def object_pointcloud(mesh_path, scale, n_points, seed=0):
    np.random.seed(seed)
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_scale(scale)
    xyz, _ = trimesh.sample.sample_surface(mesh, n_points)
    return np.asarray(xyz, dtype=np.float32)


def score_grasps(sampler, pc, grasps_4x4, batch=200):
    """Run the discriminator on GIVEN grasps (not generated ones)."""
    centre = pc.mean(axis=0)
    pts = torch.from_numpy(pc - centre[None]).float()
    g = torch.from_numpy(np.asarray(grasps_4x4, dtype=np.float32)).clone()
    g[:, :3, 3] -= torch.from_numpy(centre).float()      # same frame as the cloud

    out = []
    model = sampler.model
    for i in range(0, len(g), batch):
        chunk = g[i:i + batch]
        data = {"task": "pick",
                "inputs": torch.cat([pts, torch.zeros_like(pts)], dim=-1).float(),
                "points": pts}
        data = sampler.load_gripper_input(data)
        batch = collate([data])
        # add AFTER collate: collate cannot handle a str field, and grasps are injected the
        # same way sample() injects generator output
        batch["grasps"] = [chunk.to(_DEVICE)]
        batch["grasp_key"] = "grasps"
        with torch.inference_mode():
            res, _, _ = model.grasp_discriminator.infer(batch)
        conf = res["grasp_confidence"]
        conf = conf[0] if isinstance(conf, (list, tuple)) else conf
        out.append(conf.detach().cpu().numpy().reshape(-1)[: len(chunk)])
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gripper", default="franka_panda", choices=list(GRIPPER_MAP))
    ap.add_argument("--shard", type=int, default=6)
    ap.add_argument("--root", default=str(REPO / "data/graspgen"))
    ap.add_argument("--mesh_dir", default=str(REPO / "data/graspgen/meshes"))
    ap.add_argument("--n_objects", type=int, default=50)
    ap.add_argument("--n_grasps", type=int, default=200, help="subsample per object")
    ap.add_argument("--n_points", type=int, default=2048)
    ap.add_argument("--only_shared", action="store_true", help="restrict to today_set.json")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent / "results/auc.json"))
    args = ap.parse_args()
    logging.disable(logging.INFO)

    mmap_path = os.path.join(args.mesh_dir, "map_uuid_to_path.json")
    if not os.path.exists(mmap_path):
        raise SystemExit(f"No mesh map at {mmap_path} — run download_objaverse.py first (step 1).")
    mesh_map = json.load(open(mmap_path))
    # download_objaverse.py writes paths relative to its own cwd; resolve against mesh_dir's parent
    base = pathlib.Path(args.mesh_dir).resolve().parent
    mesh_map = {k: (v if os.path.isabs(v) else str(base / v)) for k, v in mesh_map.items()}

    uuid_filter = None
    if args.only_shared:
        uuid_filter = set(json.load(open(f"{args.root}/today_set.json"))["uuids"])

    ckpt = str(get_checkpoints_version_dir())
    cfg = load_model_cfg(os.path.join(ckpt, "gen"), os.path.join(ckpt, "dis"))
    sampler = GraspGenXSampler(cfg, GRIPPER_MAP[args.gripper],
                               assets_dir=str(REPO / "step3_grasp_graspgenx/GraspGenX/assets"))
    print(f"device={_DEVICE}  gripper={args.gripper}\n")

    shard = f"{args.root}/grasp_data/{args.gripper}/shard_{args.shard:03d}.tar"
    all_s, all_y, rows = [], [], []
    for uuid, d in iter_objects(shard, uuid_filter):
        if len(rows) >= args.n_objects:
            break
        if uuid not in mesh_map:
            continue
        T = np.array(d["grasps"]["transforms"], dtype=np.float32)
        y = np.array(d["grasps"]["object_in_gripper"], dtype=bool)
        if y.all() or not y.any():
            continue                              # AUC undefined for one-class objects
        idx = np.random.RandomState(0).choice(len(T), min(args.n_grasps, len(T)), replace=False)
        T, y = T[idx], y[idx]
        if y.all() or not y.any():
            continue          # re-check AFTER subsampling: AUC is undefined on one class
        try:
            pc = object_pointcloud(mesh_map[uuid], d["object"]["scale"], args.n_points)
            s = score_grasps(sampler, pc, T)
        except Exception as e:
            print(f"  skip {uuid[:8]}: {type(e).__name__}: {e}")
            continue
        auc = roc_auc_score(y, s)
        cat = re.sub(r".*/simplified/([^/]+)/.*", r"\1", d["object"]["file"])
        rows.append(dict(uuid=uuid, cat=cat, auc=float(auc), n=int(len(y)),
                         pos_rate=float(y.mean())))
        all_s.append(s); all_y.append(y)
        print(f"  {cat[:26]:<26} AUC {auc:.3f}   pos {y.mean():.2f}  n={len(y)}")

    if not rows:
        raise SystemExit("No objects scored — check the mesh download.")
    S, Y = np.concatenate(all_s), np.concatenate(all_y)
    pooled = roc_auc_score(Y, S)
    per_obj = float(np.nanmean([r["auc"] for r in rows]))
    print(f"\n=== {args.gripper}: {len(rows)} objects, {len(Y)} grasps ===")
    print(f"pooled AUC     {pooled:.4f}")
    print(f"mean per-object AUC {per_obj:.4f}   (the honest number — pooling mixes object difficulty)")
    verdict = ("s_geo is a good feature — build on it" if per_obj > 0.75 else
               "usable but weak — lean on property features" if per_obj > 0.6 else
               "uninformative — drop s_geo, train on the point cloud directly")
    print(f"-> {verdict}")
    json.dump(dict(gripper=args.gripper, pooled_auc=pooled, mean_object_auc=per_obj,
                   objects=rows), open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
