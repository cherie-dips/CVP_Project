#!/usr/bin/env python3
"""Step 3a. Build a labelled grasp dataset: describe each grasp by its geometry, keep its outcome.

GraspGenX's own grasp score turned out to be below chance for one of the two grippers (AUC 0.445).
Before relying on it for our fin-ray, we should know whether a simple scorer trained on the same
labels does better. This builds the features for that.

Each row is one grasp: where it sits on the object, which way it approaches, what the surface looks
like there, and whether it held.

    python step3_grasp_graspgenx/01_build_grasp_dataset.py --per_object 200
"""
import argparse, json, pathlib, re, warnings
import numpy as np
import trimesh
import webdataset as wds
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / "data/graspgen"
GRIPPERS = {"franka": "franka_panda", "robotiq": "robotiq_2f_140"}
APERTURE = {"franka": 0.080, "robotiq": 0.140}


def grasp_features(T, pts, tree, centroid, radius, aperture):
    """Geometry of one grasp relative to the object. All scale-normalised."""
    p = T[:3, 3]
    approach, closing = T[:3, 2], T[:3, 0]
    rel = (p - centroid) / radius

    d_near, idx = tree.query(p, k=32)
    near = pts[idx]
    local = near - near.mean(0)
    ev = np.linalg.eigvalsh(np.cov(local.T) + 1e-12 * np.eye(3))[::-1]
    ev = ev / (ev.sum() + 1e-12)                       # flat / edge / corner signature

    # how much of the object lies between the jaws
    v = pts - p
    along = v @ closing
    perp = np.linalg.norm(v - np.outer(along, closing), axis=1)
    in_jaw = (np.abs(along) < aperture / 2) & (perp < aperture / 4)
    # is the approach direction clear? points behind the gripper block it
    behind = ((v @ approach) < -0.005) & (np.linalg.norm(v, axis=1) < aperture)

    return np.array([
        *rel, np.linalg.norm(rel),
        *approach, *closing,
        float(d_near[0]) / radius, float(d_near.mean()) / radius,
        *ev,
        in_jaw.mean(), behind.mean(),
        float(np.abs(approach[2])),                    # 1 = straight down, 0 = from the side
    ], dtype=np.float32)


N_FEAT = 18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_object", type=int, default=200)
    ap.add_argument("--n_points", type=int, default=2048)
    ap.add_argument("--out", default=str(HERE / "results/grasp_features.npz"))
    a = ap.parse_args()

    mesh_map = {k: str(DATA / v) for k, v in
                json.load(open(DATA / "meshes/map_uuid_to_path.json")).items()}
    idx = {g: json.load(open(DATA / f"grasp_data/{d}/uuid_index.json"))
           for g, d in GRIPPERS.items()}
    have = set(mesh_map) & set(idx["franka"]) & set(idx["robotiq"])
    print(f"{len(have)} objects have a mesh and both grippers' grasps")

    clouds = {}
    rng = np.random.default_rng(0)
    X, y, obj, grip = [], [], [], []

    for g, dname in GRIPPERS.items():
        shards = sorted({idx[g][u] for u in have})
        for s in shards:
            p = DATA / f"grasp_data/{dname}/shard_{s:03d}.tar"
            if not p.exists():
                continue
            for sample in wds.WebDataset(str(p), shardshuffle=False):
                u = sample["__key__"]
                if u not in have:
                    continue
                j = json.loads(sample["grasps.json"])
                if u not in clouds:
                    try:
                        m = trimesh.load(mesh_map[u], force="mesh")
                        m.apply_scale(j["object"]["scale"])
                        np.random.seed(0)
                        pts, _ = trimesh.sample.sample_surface(m, a.n_points)
                        pts = np.asarray(pts, np.float32)
                        clouds[u] = (pts, cKDTree(pts), pts.mean(0),
                                     float(np.linalg.norm(pts - pts.mean(0), axis=1).max()),
                                     re.sub(r".*/simplified/([^/]+)/.*", r"\1", j["object"]["file"]))
                    except Exception:
                        clouds[u] = None
                if clouds[u] is None:
                    continue
                pts, tree, cen, rad, cat = clouds[u]
                T = np.asarray(j["grasps"]["transforms"], np.float32)
                lab = np.asarray(j["grasps"]["object_in_gripper"], bool)
                k = min(a.per_object, len(T))
                sel = rng.choice(len(T), k, replace=False)
                for i in sel:
                    X.append(grasp_features(T[i], pts, tree, cen, rad, APERTURE[g]))
                    y.append(lab[i]); obj.append(u); grip.append(g)
            print(f"  {dname} shard {s}: {len(X)} grasps", flush=True)

    X = np.stack(X); y = np.array(y, np.int8)
    np.savez_compressed(a.out, X=X, y=y, obj=np.array(obj), grip=np.array(grip),
                        cat=np.array([clouds[o][4] for o in obj]))
    print(f"\n{len(X)} grasps over {len(set(obj))} objects   success rate {y.mean():.3f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
