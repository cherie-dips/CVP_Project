"""Old and fixed grasp planning for both gripper configurations, checked against the real gripper.

The first evaluation measured "standoff": whether the object crosses the top of the swept box. That
box (params_legacy.json, 148 x 30 x 38 mm) does not match the STLs -- the real fingers form a V
that cuts through it and reach 26 mm past its top -- and standoff never asked whether the fingers
touch the object or whether the grip would hold. Every grasp here is instead closed with the real
fingers (finray.Checker): no collision when open, every finger touches before they meet, the two
pairs grip within the fin-ray's bending allowance, and each pair squeezes along its contact line.

    2f legacy      GraspGenX diffusion, the old 2-finger box                  (the original pipeline)
    2f stl         GraspGenX diffusion, the 2-finger box GraspGenX's own wizard convention gives
                   for the STLs: 66 x 10 x 85 mm (the gap at the finger roots; the tips are 148 apart)
    2f tuned       GraspGenX diffusion, the box tune_gripper_box.py chose (data/gripper/params_tuned.json)
    2f tuned moe   GraspGenX's default planner (GraspMoE: diffusion + bounding-box sweep), tuned box
    4f legacy      GraspGenX diffusion, the old square 4-finger box            (the original pipeline)
    4f stl box     GraspGenX diffusion, the square 4-finger box from the STLs (right numbers, same
                   representation -- separates "wrong numbers" from "a box cannot say four fingers")
    4f composed, no rack check
                   four_finger.plan with the chosen 2-finger box, one_rack=False: two 2-finger
                   pinches at 90 degrees, both scored by GraspGenX's discriminator, nothing else
    4f composed    the same, plus the one-rack width check, with the "4f stl box" grasps as extra
                   candidates -- the fixed 4-finger planner

Objects: the 50 real-scanned objects (26 YCB, 24 Google Scanned Objects) that are stage 1's
external test set, so each one also has the stage 1 label (does the rule say four fingers?).

    /opt/anaconda3/envs/graspgenx/bin/python stage2_grasp/scripts/evaluate_grasps.py [--only m1,m2]
        -> stage2_grasp/results/grasp_check.csv
           data/grasp_eval/<object>.npz (every pose, score and check, for re-plotting)
        --only re-runs just those methods and merges them into the existing results
"""
import json, os, pathlib, sys, time, warnings
os.environ["GRASPGENX_DEVICE"] = "cpu"                 # MPS gives wrong numbers for this model
os.environ["PXR_USDC_EMIT_DEPRECATION_WARNINGS"] = "0"

import numpy as np, pandas as pd, torch, trimesh

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"stage2_grasp"/"scripts"))
import finray, four_finger, objects
from graspgenx import get_checkpoints_version_dir
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.samplers.graspmoe import run_graspmoe
from graspgenx.utils.checkpoint_io import load_model_cfg
from graspgenx.x_grippers import make_sweep_volume_gripper_info

OUT = ROOT/"stage2_grasp"/"results"
RAW = ROOT/"data"/"grasp_eval"
TOP = 10                    # grasps checked per method: the 10 the planner scores highest
POOL = 100                  # for "is there any valid grasp": the 100 highest
warnings.filterwarnings("ignore")


def stl_boxes(n_fingers):
    """GraspGenX's wizard convention (scripts/gripper_config_wizard.py, estimate_inner_sweep_volume):
    along each closing axis, the gap between the innermost finger surfaces; along the others, the
    union of the fingers' bounding boxes. Open at THETA_OPEN, half-open halfway to closed."""
    from assemble_gripper import FING

    def box(theta):
        P = []
        for a in finray.abouts(n_fingers):
            T = finray.finger_pose(theta, a)
            P.append(FING.vertices * 1e-3 @ T[:3, :3].T + T[:3, 3])
        lo = np.min([p.min(0) for p in P], 0); hi = np.max([p.max(0) for p in P], 0)
        by = dict(zip(finray.abouts(n_fingers), P))
        lo[0], hi[0] = by[180][:, 0].max(), by[0][:, 0].min()
        if n_fingers == 4:
            lo[1], hi[1] = by[270][:, 1].max(), by[90][:, 1].min()
        return (hi - lo).round(4).tolist(), ((hi + lo) / 2).round(4).tolist()

    (eo, oo), (em, om) = box(finray.THETA_OPEN), box((finray.THETA_OPEN + finray.THETA_CLOSED) / 2)
    return dict(extents_open=eo, offset_open=oo, extents_mid=em, offset_mid=om)


def catalogue():
    df = pd.read_csv(ROOT/"stage1_gripper"/"results"/"selector"/"objects.csv")
    return df[df.split == "external"].reset_index(drop=True)


def load(row):
    if row.source == "ycb":
        return objects.ycb(row.id[4:])
    return objects.usd(ROOT/"selected_objects"/row.id[4:]/f"{row.id[4:]}.usd")


def diffusion(pts, sampler):
    poses, conf = GraspGenXSampler.run_inference(pts, sampler, grasp_threshold=-1.0, num_grasps=200,
                                                 topk_num_grasps=POOL, remove_outliers=False)
    poses, conf = poses.cpu().numpy(), conf.cpu().numpy()
    order = np.argsort(-conf)
    return poses[order], conf[order]


def moe(pts, sampler):
    m = run_graspmoe(pts, sampler, num_grasps=200, topk_num_grasps=-1)
    poses = np.concatenate([m["grasps_diff"], m["grasps_obb"]])
    conf = np.concatenate([m["scores_diff"], m["scores_obb"]])
    order = np.argsort(-conf)[:POOL]
    return poses[order], conf[order]


def main():
    OUT.mkdir(exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    legacy = json.loads((ROOT/"data"/"gripper"/"params_legacy.json").read_text())["grippers"]
    tuned = json.loads((ROOT/"data"/"gripper"/"params_tuned.json").read_text())["grippers"]["2-finger"]
    boxes = {"legacy 2": legacy["2-finger"], "legacy 4": legacy["4-finger"], "stl 2": stl_boxes(2),
             "stl 4": stl_boxes(4), "tuned 2": tuned}
    (OUT/"gripper_boxes.json").write_text(json.dumps(
        {"theta_open_deg": round(finray.THETA_OPEN, 2), "theta_closed_deg": finray.THETA_CLOSED,
         "tip_gap_open_mm": finray.TIP_GAP_OPEN,
         **{k: {kk: v[kk] for kk in ("extents_open", "offset_open", "extents_mid", "offset_mid")}
            for k, v in boxes.items()}}, indent=1))
    ckpt = get_checkpoints_version_dir()
    cfg = load_model_cfg(str(ckpt/"gen"), str(ckpt/"dis"))
    model, samplers = None, {}
    for k, g in boxes.items():
        info = make_sweep_volume_gripper_info(g["extents_open"], g["offset_open"], g["extents_mid"],
                                              g["offset_mid"], name=k.replace(" ", "_"))
        samplers[k] = GraspGenXSampler(cfg, gripper_info=info, model=model)
        model = samplers[k].model

    methods = {
        "2f legacy": (2, lambda pts: diffusion(pts, samplers["legacy 2"])),
        "2f stl": (2, lambda pts: diffusion(pts, samplers["stl 2"])),
        "2f tuned": (2, lambda pts: diffusion(pts, samplers["tuned 2"])),
        "2f tuned moe": (2, lambda pts: moe(pts, samplers["tuned 2"])),
        "4f legacy": (4, lambda pts: diffusion(pts, samplers["legacy 4"])),
        "4f stl box": (4, lambda pts: diffusion(pts, samplers["stl 4"])),
        "4f composed, no rack check": (4, lambda pts: four_finger.plan(
            pts, samplers["tuned 2"], topk=POOL, one_rack=False)[:2]),
        "4f composed": (4, lambda pts: four_finger.plan(
            pts, samplers["tuned 2"], topk=POOL, extra_candidates=raw["4f stl box"]["poses"])[:2]),
    }
    if all(np.allclose(tuned[k], legacy["2-finger"][k]) for k in ("extents_open", "offset_open", "extents_mid", "offset_mid")):
        del methods["2f tuned"]                     # tuning kept the old box: same as "2f legacy"
        print("tuning kept the old 2-finger box; the tuned methods use it")

    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    if only:
        methods = {k: v for k, v in methods.items() if k in only}
    partial = RAW/("grasp_check_partial.csv" if not only else f"grasp_check_partial_{len(only)}.csv")
    rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []      # resume
    done = {r["id"] for r in rows}
    for obj in catalogue().itertuples():
        if obj.id in done:
            continue
        old_raw = {}
        if only and (RAW/f"{obj.id}.npz").exists():                  # keep the other methods' arrays
            z = np.load(RAW/f"{obj.id}.npz", allow_pickle=True)
            for key in z.files:
                if "|" in key:
                    m, kk = key.split("|")
                    old_raw.setdefault(m, {})[kk] = z[key]
        mesh = load(obj)
        pts = np.asarray(trimesh.sample.sample_surface(mesh, 3500, seed=0)[0], np.float32)
        centre = pts.mean(0)
        pts -= centre                                       # the model expects a centred cloud
        mesh.apply_translation(-centre)
        checkers = {n: finray.Checker(mesh, n) for n in (2, 4)}
        raw = dict(old_raw)
        for name, (n, plan) in methods.items():
            np.random.seed(0); torch.manual_seed(0)
            t = time.time()
            poses, conf = plan(pts)
            t_plan = time.time() - t
            checks = [checkers[n].check(P) for P in poses]
            first_valid = next((i for i, c in enumerate(checks) if c["valid"]), None)
            top = checks[:TOP]
            rows.append(dict(
                object=obj.name, id=obj.id, source=obj.source, rule=obj.label, circularity=obj.circularity,
                short_mm=obj.short_mm, method=name, fingers=n, seconds=round(t_plan, 1),
                top_score=float(conf[0]) if len(conf) else np.nan,
                reach_top10=np.mean([c["reaches"] for c in top]) if top else 0.0,
                valid_top10=np.mean([c["valid"] for c in top]) if top else 0.0,
                collides_open_top10=np.mean([c["collides_open"] for c in top]) if top else 0.0,
                top1_valid=bool(checks and checks[0]["valid"]),
                any_valid=first_valid is not None, first_valid_rank=first_valid,
                why_top1_fails=checks[0]["reason"] if checks and not checks[0]["valid"] else ""))
            raw[name] = dict(poses=poses, conf=conf, valid=np.array([c["valid"] for c in checks]),
                             reaches=np.array([c["reaches"] for c in checks]))
        np.savez_compressed(RAW/f"{obj.id}.npz", centre=centre,
                            **{f"{k}|{kk}": v for k, d in raw.items() for kk, v in d.items()})
        pd.DataFrame(rows).to_csv(partial, index=False)                              # after every object
        line = "  ".join(f"{r['method']} {r['valid_top10']:.0%}" for r in rows[-len(methods):])
        print(f"{obj.name[:24]:24s} ({obj.label})  {line}", flush=True)

    df = pd.DataFrame(rows)
    if only and (OUT/"grasp_check.csv").exists():                       # merge into the full table
        full = pd.read_csv(OUT/"grasp_check.csv")
        df = pd.concat([full[~full.method.isin(only)], df], ignore_index=True)
    df.to_csv(OUT/"grasp_check.csv", index=False, float_format="%.3f")
    summary(df)


def summary(df):
    g = df.groupby(["method", "rule"]).agg(objects=("object", "count"), valid_top10=("valid_top10", "mean"),
                                           reach_top10=("reach_top10", "mean"), top1_valid=("top1_valid", "mean"),
                                           any_valid=("any_valid", "mean"), seconds=("seconds", "median"))
    with pd.option_context("display.width", 160, "display.float_format", "{:.2f}".format):
        print(g.to_string())


if __name__ == "__main__":
    main()
