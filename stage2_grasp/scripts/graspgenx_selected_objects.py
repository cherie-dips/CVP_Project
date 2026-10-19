"""GraspGenX on the objects in selected_objects/, with both the 2-finger and the 4-finger gripper.

The same pipeline as stage2_grasp.ipynb and stage2_grasp_4finger.ipynb, run on the Isaac Sim USD
assets in selected_objects/ (5 YCB objects and 24 Google Scanned Objects) instead of the 10 YCB
.ply meshes. Writes, to stage2_grasp/results/:

    graspgenx_selected_objects.png / .csv       2-finger collage and scores
    graspgenx_selected_objects_4f.png / .csv    4-finger collage, standoff vs the 2-finger

Run from anywhere:  /opt/anaconda3/envs/graspgenx/bin/python stage2_grasp/scripts/graspgenx_selected_objects.py
"""
import argparse, json, os, pathlib, time
os.environ["GRASPGENX_DEVICE"] = "cpu"                 # MPS gives wrong numbers for this model
os.environ["PXR_USDC_EMIT_DEPRECATION_WARNINGS"] = "0"  # the assets are crate 0.7.0

import numpy as np, pandas as pd, torch, trimesh, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pxr import Usd, UsdGeom

from graspgenx import get_checkpoints_version_dir
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.utils.checkpoint_io import load_model_cfg
from graspgenx.x_grippers import make_sweep_volume_gripper_info

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT  = ROOT/"stage2_grasp"/"results"; OUT.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--objects", default=ROOT/"selected_objects", type=pathlib.Path)
ap.add_argument("--params",  default=ROOT/"data"/"gripper"/"params_legacy.json", type=pathlib.Path)
ap.add_argument("--seed",    default=0, type=int)
args = ap.parse_args()
np.random.seed(args.seed); torch.manual_seed(args.seed)

# Google Scanned Objects names are product listings; these are what the collage titles show.
SHORT = {
    "ACE_Coffee_Mug_Kristen_16_oz_cup": "coffee mug",
    "BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup": "porcelain ramekin",
    "Big_Dot_Aqua_Pencil_Case": "aqua pencil case",
    "Calphalon_Kitchen_Essentials_12_Cast_Iron_Fry_Pan_Black": "cast-iron fry pan",
    "Central_Garden_Flower_Pot_Goo_425": "flower pot",
    "Chefmate_8_Frypan": "8 in frypan",
    "Cole_Hardware_Dishtowel_Blue": "blue dishtowel",
    "Craftsman_Grip_Screwdriver_Phillips_Cushion": "screwdriver",
    "Crayola_Bonus_64_Crayons": "crayon box",
    "Diamond_Visions_Scissors_Red": "red scissors",
    "Ecoforms_Cup_B4_SAN": "plant cup",
    "Horses_in_Pink_Pencil_Case": "pink pencil case",
    "Now_Designs_Bowl_Akita_Black": "black bowl",
    "Now_Designs_Dish_Towel_Mojave_18_x_28": "mojave dish towel",
    "Olive_Kids_Butterfly_Garden_Pencil_Case": "butterfly pencil case",
    "Pennington_Electric_Pot_Cabana_4": "cabana pot",
    "Remington_TStudio_Hair_Dryer": "hair dryer",
    "Room_Essentials_Bowl_Turquiose": "turquoise bowl",
    "Sapota_Threshold_4_Ceramic_Round_Planter_Red": "red planter",
    "Sea_to_Summit_Xl_Bowl": "camping bowl",
    "Tag_Dishtowel_18_x_26": "tag dishtowel",
    "Tag_Dishtowel_Green": "green dishtowel",
    "Threshold_Bead_Cereal_Bowl_White": "cereal bowl",
    "Threshold_Porcelain_Serving_Bowl_Coupe_White": "serving bowl",
}

def label(name):
    if name in SHORT:
        return SHORT[name]
    head, _, rest = name.partition("_")
    return (rest if head.isdigit() else name).replace("_", " ")   # YCB: 011_banana -> banana


OBJECTS = sorted(p.name for p in args.objects.iterdir() if (p/f"{p.name}.usd").exists())
print(len(OBJECTS), "objects")

# ---- 1. are these objects really unseen? --------------------------------------------------
# GraspGenX trains on Objaverse, indexed by UUID; these are YCB and Google Scanned Objects.
seen = set()
for f in (ROOT/"data"/"objaverse"/"splits").glob("*/valid.txt"):
    seen |= {l.strip() for l in open(f) if l.strip()}
print(f"{len(seen)} object ids in the GraspGen splits, overlap with ours: {seen & set(OBJECTS)}")

# ---- 2. load the USD assets ---------------------------------------------------------------

def load_usd(path):
    """The visual mesh of an Isaac Sim asset, in metres, with its prim transforms applied."""
    stage = Usd.Stage.Open(str(path))
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    xform = UsdGeom.XformCache()
    parts = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputePurpose() != UsdGeom.Tokens.default_:
            continue                                    # skip the collision copy (purpose=guide)
        m = UsdGeom.Mesh(prim)
        v = np.asarray(m.GetPointsAttr().Get(), float)
        T = np.asarray(xform.GetLocalToWorldTransform(prim))   # row-vector convention
        v = (v @ T[:3, :3] + T[3, :3]) * scale
        counts = np.asarray(m.GetFaceVertexCountsAttr().Get())
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get())
        faces, k = [], 0
        for c in counts:                                # fan-triangulate any n-gons
            faces += [[idx[k], idx[k+j], idx[k+j+1]] for j in range(1, c-1)]
            k += c
        parts.append(trimesh.Trimesh(v, np.array(faces), process=False))
    return trimesh.util.concatenate(parts)

# ---- 3. the two grippers ------------------------------------------------------------------
params = json.load(open(args.params))
grippers = {}
for key, tag in (("2-finger", "finray_2f"), ("4-finger", "finray_4f")):
    g = params["grippers"][key]
    grippers[key] = make_sweep_volume_gripper_info(
        g["extents_open"], g["offset_open"], g["extents_mid"], g["offset_mid"], name=tag)

g4 = params["grippers"]["4-finger"]
REACH  = g4["extents_open"][0]/2                            # centre to each finger
Z_TIP  = g4["offset_open"][2] + g4["extents_open"][2]/2     # front of the swept box
Z_BACK = g4["offset_open"][2] - g4["extents_open"][2]/2     # back of it
FINGERS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], float)
assert params["grippers"]["2-finger"]["offset_open"][2] == g4["offset_open"][2]   # same fingers, same depth
print(f"params {args.params.name}: aperture {g4['extents_open'][0]*1000:.1f} mm, "
      f"fingers at z = {Z_BACK*1000:.1f}..{Z_TIP*1000:.1f} mm")

# ---- 4. the model -------------------------------------------------------------------------
ckpt = get_checkpoints_version_dir()
cfg  = load_model_cfg(str(ckpt/"gen"), str(ckpt/"dis"))
samplers = {"2-finger": GraspGenXSampler(cfg, gripper_info=grippers["2-finger"])}
samplers["4-finger"] = GraspGenXSampler(cfg, gripper_info=grippers["4-finger"],
                                        model=samplers["2-finger"].model)

# ---- 5. inference -------------------------------------------------------------------------

def standoff_mm(pose, dense):
    """How far past the fingertips the object begins (mm). <= 0 means the fingers reach it."""
    Ti = np.linalg.inv(pose)
    p = dense @ Ti[:3, :3].T + Ti[:3, 3]                # object seen from the gripper
    return (p[:, 2].min() - Z_TIP) * 1000


results = {}
for name in OBJECTS:
    t = time.time()
    mesh = load_usd(args.objects/name/f"{name}.usd")
    pts = np.asarray(trimesh.sample.sample_surface(mesh, 3500)[0], dtype=np.float32)
    centre = pts.mean(0)
    pts -= centre                                       # the model expects a centred cloud
    mesh.apply_translation(-centre)                     # keep the mesh in the same frame
    dense = np.asarray(trimesh.sample.sample_surface(mesh, 40000)[0])

    r = results[name] = dict(pts=pts, dense=dense, size=mesh.extents*1000)
    for key, sampler in samplers.items():
        poses, conf = GraspGenXSampler.run_inference(
            pts, sampler, grasp_threshold=-1.0, num_grasps=200,
            topk_num_grasps=20, remove_outliers=False)
        poses, conf = poses.cpu().numpy(), conf.cpu().numpy()
        r[key] = dict(poses=poses, conf=conf,
                      standoff=np.array([standoff_mm(T, dense) for T in poses]))
    print(f"{label(name):22s} {r['size'].min():4.0f} mm   "
          f"2f score {r['2-finger']['conf'].max():.2f} reach {np.mean(r['2-finger']['standoff'] <= 0):4.0%}   "
          f"4f score {r['4-finger']['conf'].max():.2f} reach {np.mean(r['4-finger']['standoff'] <= 0):4.0%}   "
          f"{time.time()-t:.1f}s")

# ---- 6. collages --------------------------------------------------------------------------
COLS = 6
ROWS = -(-len(OBJECTS) // COLS)
BLUE, ORANGE = "#1f6fd0", "#e8a33d"


def fork(pose):
    """The 2-finger outline (7 points) placed at one grasp pose."""
    pts = np.array(grippers["2-finger"].control_points_visualization[0], dtype=float)
    return [(pose @ pts.T).T[:, :3]]


def wireframe(pose):
    """The four fingers, plus the stem back to the gripper base."""
    segs = [[[0, 0, Z_BACK], [0, 0, 0]]]                       # stem
    for fx, fy in FINGERS:
        segs.append([[0, 0, Z_BACK],
                     [REACH*fx, REACH*fy, Z_BACK],             # spoke out to the knuckle
                     [REACH*fx, REACH*fy, Z_TIP]])             # the finger itself
    return [(pose @ np.c_[np.array(s), np.ones(len(s))].T).T[:, :3] for s in segs]


def four_contacts(pose, dense):
    """Where each of the four fingers would touch, for grasps that do reach the object."""
    Ti = np.linalg.inv(pose)
    p = dense @ Ti[:3, :3].T + Ti[:3, 3]
    band = (p[:, 2] >= Z_BACK) & (p[:, 2] <= Z_TIP)
    if band.sum() < 10:
        return None
    q = p[band]
    contacts = [q[int(np.argmax(q[:, 0]*fx + q[:, 1]*fy))] for fx, fy in FINGERS]
    return np.array(contacts) @ pose[:3, :3].T + pose[:3, 3]


def collage(key, n_show, draw, title, suptitle, path, lw_best, lw_rest, contacts=False):
    fig = plt.figure(figsize=(3.0*COLS, 3.5*ROWS))
    for i, name in enumerate(OBJECTS):
        r, g = results[name], results[name][key]
        top = np.argsort(-g["conf"])[:n_show]

        ax = fig.add_subplot(ROWS, COLS, i+1, projection="3d")
        ax.scatter(*r["pts"][::2].T, s=2.5, c="0.6", alpha=.55, linewidths=0)
        for rank, k in enumerate(reversed(top)):
            best = rank == len(top)-1
            for seg in draw(g["poses"][k]):
                ax.plot(*seg.T, color=BLUE if best else ORANGE,
                        lw=lw_best if best else lw_rest, alpha=1.0 if best else .8,
                        zorder=3 if best else 2)
        if contacts:
            c = four_contacts(g["poses"][top[0]], r["dense"])
            if c is not None:
                ax.scatter(*c.T, s=34, c="#c0392b", edgecolors="white", lw=.6, zorder=5)

        # frame both the object and the winning gripper pose; a radius, not the largest coordinate,
        # so a gripper pointing at a corner of the box is not clipped at the panel edge
        radius = lambda p: np.linalg.norm(p, axis=1).max()
        lim = max(radius(r["pts"]), max(radius(s) for s in draw(g["poses"][top[0]])))
        for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim): setter(-lim, lim)
        ax.set_box_aspect((1, 1, 1), zoom=1.5)
        ax.view_init(elev=20, azim=45); ax.set_axis_off()
        ax.set_title(title(name, r, g, top[0]), fontsize=9.5, pad=-4)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=.995)
    fig.subplots_adjust(top=1 - .22/ROWS, bottom=.01, left=.01, right=.99, wspace=0, hspace=.18)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print("saved", path.relative_to(ROOT))


N = len(OBJECTS)
collage("2-finger", 3, fork,
        lambda name, r, g, k: f"{label(name)}\n{r['size'].min():.0f} mm  ·  top score {g['conf'].max():.2f}",
        f"GraspGenX with our fin-ray gripper, on {N} YCB and Google Scanned objects it was never trained on",
        OUT/"graspgenx_selected_objects.png", lw_best=3.0, lw_rest=1.7)
collage("4-finger", 2, wireframe,
        lambda name, r, g, k: f"{label(name)}\n{r['size'].min():.0f} mm  ·  score {g['conf'][k]:.2f}  ·  "
                              f"standoff {g['standoff'][k]:.0f} mm",
        f"GraspGenX with our 4-finger fin-ray gripper, on {N} YCB and Google Scanned objects it never saw",
        OUT/"graspgenx_selected_objects_4f.png", lw_best=2.6, lw_rest=1.5, contacts=True)

# ---- 7. tables ----------------------------------------------------------------------------
pd.DataFrame([
    dict(object=label(n),
         short_side_mm=round(results[n]["size"].min()),
         grasps=len(results[n]["2-finger"]["poses"]),
         top_score=round(float(results[n]["2-finger"]["conf"].max()), 3),
         median_score=round(float(np.median(results[n]["2-finger"]["conf"])), 3))
    for n in OBJECTS]).to_csv(OUT/"graspgenx_selected_objects.csv", index=False)

table = pd.DataFrame([
    dict(object=label(n),
         short_mm=round(results[n]["size"].min()),
         standoff_2f_mm=round(float(np.median(results[n]["2-finger"]["standoff"])), 1),
         reach_2f=round(float(np.mean(results[n]["2-finger"]["standoff"] <= 0)), 2),
         standoff_4f_mm=round(float(np.median(results[n]["4-finger"]["standoff"])), 1),
         reach_4f=round(float(np.mean(results[n]["4-finger"]["standoff"] <= 0)), 2))
    for n in OBJECTS])
table.to_csv(OUT/"graspgenx_selected_objects_4f.csv", index=False)

print(f"\nmedian standoff — 2-finger {table.standoff_2f_mm.median():.1f} mm, "
      f"4-finger {table.standoff_4f_mm.median():.1f} mm")
print(f"grasps that reach the object — 2-finger {table.reach_2f.mean():.0%}, "
      f"4-finger {table.reach_4f.mean():.0%}")
