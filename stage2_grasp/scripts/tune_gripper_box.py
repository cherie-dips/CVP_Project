"""Choose the 12 numbers GraspGenX is told about our 2-finger gripper, by checking what it plans.

A swept box is a poor fit for our fingers: they form a V when open, 66 mm apart at the root and
148 mm at the tips, and the box has one width. GraspGenX's wizard convention (gap between the
fingers' bounding boxes) gives 66 mm and makes the model avoid anything wider; the old box claims
148 mm over the whole finger band. So treat the box as what it is, a description to calibrate:
for each candidate box, plan 2-finger grasps on Objaverse objects from stage 1's training split
and close the real fingers on them (finray.Checker). Keep the box whose top-10 grasps are valid
most often. The 50 objects stage 2 is evaluated on are not used here.

    width       box x when open: 66, 88, 110, 132, 148 mm
    band        "fingertips": the old box's z ranges (76-114 mm above the mounting face open) and
                its half-open width ratio, 0.69
                "whole finger": the fingers' full z range (53-138 mm) and the wizard's ratio, 0.78
                -- so (148, fingertips, 30) is the old box exactly
    thickness   10 mm, the finger's real thickness (the old box used 30 mm, and is run as well)

A candidate replaces the old box only if it beats it on the same objects by a paired Wilcoxon
signed-rank test at p < 0.05; a difference the 16 objects cannot resolve keeps the old box.

    /opt/anaconda3/envs/graspgenx/bin/python stage2_grasp/scripts/tune_gripper_box.py [--select]
        -> stage2_grasp/results/gripper_box_tuning.csv, data/gripper/params_tuned.json
        (--select: only re-read the CSV and choose)
"""
import itertools, json, os, pathlib, sys, time, warnings
os.environ["GRASPGENX_DEVICE"] = "cpu"

import numpy as np, pandas as pd, torch, trimesh

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"stage2_grasp"/"scripts"))
import finray, objects
from graspgenx import get_checkpoints_version_dir
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.utils.checkpoint_io import load_model_cfg
from graspgenx.x_grippers import make_sweep_volume_gripper_info

warnings.filterwarnings("ignore")
WIDTHS = [0.066, 0.088, 0.110, 0.132, 0.148]
BANDS = {  # z centre and extent open, the same half-open, half-open width / open width
    "fingertips": (0.0949, 0.0383, 0.1006, 0.0446, 0.687),
    "whole finger": (0.0956, 0.0854, 0.1048, 0.0856, 0.780),
}
N_OBJECTS = 16


def box(width, band, thick):
    z, ez, zm, ezm, ratio = BANDS[band]
    return dict(extents_open=[width, thick, ez], offset_open=[0.0, 0.0, z],
                extents_mid=[round(ratio * width, 4), thick, ezm], offset_mid=[0.0, 0.0, zm])


def tuning_objects():
    """Objaverse training objects small enough to grasp, half of them labelled 4-finger."""
    df = pd.read_csv(ROOT/"stage1_gripper"/"results"/"selector"/"objects.csv")
    df = df[(df.split == "train") & (df.short_mm < 120) & (df.faces < 200000)]
    rng = np.random.default_rng(0)
    pick = [rng.choice(df[df.label == lab].id.values, N_OBJECTS // 2, replace=False) for lab in ("2-finger", "4-finger")]
    return list(np.concatenate(pick))


def main():
    ckpt = get_checkpoints_version_dir()
    cfg = load_model_cfg(str(ckpt/"gen"), str(ckpt/"dis"))
    configs = list(itertools.product(WIDTHS, BANDS, [0.010])) + [(0.148, "fingertips", 0.030)]   # + the old box
    samplers, model = [], None
    for w, b, t in configs:
        g = box(w, b, t)
        samplers.append(GraspGenXSampler(cfg, model=model, gripper_info=make_sweep_volume_gripper_info(
            g["extents_open"], g["offset_open"], g["extents_mid"], g["offset_mid"], name="tune")))
        model = samplers[-1].model

    out_csv = ROOT/"stage2_grasp"/"results"/"gripper_box_tuning.csv"
    rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []     # resume
    done = {r["object"] for r in rows}
    for uid in tuning_objects():
        if uid in done:
            continue
        t0 = time.time()
        mesh = objects.objaverse(uid)
        pts = np.asarray(trimesh.sample.sample_surface(mesh, 3500, seed=0)[0], np.float32)
        c = pts.mean(0); pts -= c; mesh.apply_translation(-c)
        check = finray.Checker(mesh, 2)
        for (w, b, t), s in zip(configs, samplers):
            np.random.seed(0); torch.manual_seed(0)
            P, C = GraspGenXSampler.run_inference(pts, s, grasp_threshold=-1.0, num_grasps=200,
                                                  topk_num_grasps=40, remove_outliers=False)   # one pass
            P = P.cpu().numpy()[np.argsort(-C.cpu().numpy())[:10]]
            r = [check.check(p) for p in P]
            rows.append(dict(object=uid, width_mm=round(w * 1000), band=b, thickness_mm=round(t * 1000),
                             valid_top10=np.mean([x["valid"] for x in r]), reach_top10=np.mean([x["reaches"] for x in r])))
        pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.3f")      # after every object
        print(uid, f"{time.time() - t0:.0f}s, best so far: "
              f"{pd.DataFrame(rows).groupby(['width_mm', 'band', 'thickness_mm']).valid_top10.mean().max():.2f}", flush=True)

    select()


def select():
    from scipy.stats import wilcoxon
    df = pd.read_csv(ROOT/"stage2_grasp"/"results"/"gripper_box_tuning.csv")
    key = ["width_mm", "band", "thickness_mm"]
    g = df.groupby(key).agg(valid_top10=("valid_top10", "mean"), reach_top10=("reach_top10", "mean")).reset_index()
    g = g.sort_values("valid_top10", ascending=False).reset_index(drop=True)
    print(g.to_string(index=False))
    per = df.pivot_table(index="object", columns=key, values="valid_top10")
    old = per[(148, "fingertips", 30)]
    best = tuple(g.iloc[0][key])
    p = 1.0 if best == (148, "fingertips", 30) or (per[best] == old).all() else \
        wilcoxon(per[best], old, alternative="greater").pvalue
    chosen = best if p < 0.05 else (148, "fingertips", 30)
    print(f"best {best}: {g.iloc[0].valid_top10:.3f} vs the old box {old.mean():.3f} "
          f"(paired Wilcoxon p = {p:.3f}) -> {'new box' if chosen != (148, 'fingertips', 30) else 'keep the old box'}")
    w, band, t = chosen
    legacy = json.loads((ROOT/"data"/"gripper"/"params_legacy.json").read_text())["grippers"]["2-finger"]
    box_ = legacy if chosen == (148, "fingertips", 30) else {**box(w / 1000, band, t / 1000), "n_contacts": 2}
    (ROOT/"data"/"gripper"/"params_tuned.json").write_text(json.dumps({
        "source": "stage2_grasp/scripts/tune_gripper_box.py: of 11 candidate boxes, the one whose GraspGenX "
                  "grasps the real fingers (finray.py) close on validly most often on "
                  f"{df.object.nunique()} Objaverse training objects, if it beats the old box significantly",
        "best_candidate": dict(zip(key, map(lambda v: v.item() if hasattr(v, "item") else v, best))),
        "best_valid_top10": round(float(g.iloc[0].valid_top10), 3),
        "old_box_valid_top10": round(float(old.mean()), 3),
        "paired_wilcoxon_p": round(float(p), 4),
        "chosen": "old box" if chosen == (148, "fingertips", 30) else "best candidate",
        "grippers": {"2-finger": box_}}, indent=1))


if __name__ == "__main__":
    select() if "--select" in sys.argv else main()
