"""GraspGenX grasps for the physics simulation (sim_grasps.py): 8 Objaverse test objects, both grippers.

    4-finger  four_finger.plan with the 2-finger box and the "4f stl box" diffusion grasps as extra
              candidates, one-rack check on ("4f composed" in evaluate_grasps.py)
    2-finger  GraspGenX diffusion with the same 2-finger box ("2f legacy")

The 10 grasps each planner scores highest are kept in its order, with finray.Checker's verdict on
each. The centred object mesh is written next to them, so the simulation, which does not need
GraspGenX, loads exactly the geometry the grasps were planned on.

    /opt/anaconda3/envs/graspgenx/bin/python stage2_grasp/scripts/sim_plan.py
        -> data/sim/<uid>.npz, data/sim/objects/<uid>.obj, data/sim/objects.json
"""
import json, os, pathlib, sys, time, warnings
os.environ["GRASPGENX_DEVICE"] = "cpu"                 # MPS gives wrong numbers for this model

import numpy as np, pandas as pd, torch, trimesh

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"stage2_grasp"/"scripts"))
import finray, four_finger, objects
from evaluate_grasps import diffusion, stl_boxes
from graspgenx import get_checkpoints_version_dir
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.utils.checkpoint_io import load_model_cfg
from graspgenx.x_grippers import make_sweep_volume_gripper_info

warnings.filterwarnings("ignore")
SIM = ROOT/"data"/"sim"
TOP = 10                    # grasps simulated per object and gripper, in planner order
POOL = 100

# 8 Objaverse objects from stage 1's test split, all narrower than the 148 mm opening:
# 4 the stage 1 rule sends to 4 fingers (round cross-section), 4 it sends to 2 (flat-sided)
OBJECTS = {
    "f0296ef0a4604b8195f7361320bfaa7a": "bottle",
    "b302d2bbcc3d46ba80f6ab808fe95b5a": "jack-o-lantern",
    "b5097eeb96b34ef3a3ead5f2561043fe": "light bulb",
    "60ea557999934e4ca5d513d99d8f8b70": "chess pawn",
    "7a89051127924a2ba793a72d092abe74": "cigarette pack",
    "f32ec9229a8749d1b79245813fbd6c31": "toy fridge",
    "b11b66f233df4818a70f7f30919c94e8": "cow figurine",
    "ab51eebd3c3d447d9e2968efd3085bce": "spam musubi",
}


def main():
    (SIM/"objects").mkdir(parents=True, exist_ok=True)
    cat = pd.read_csv(ROOT/"stage1_gripper"/"results"/"selector"/"objects.csv").set_index("id")
    tuned = json.loads((ROOT/"data"/"gripper"/"params_tuned.json").read_text())["grippers"]["2-finger"]
    boxes = {"tuned 2": tuned, "stl 4": stl_boxes(4)}
    ckpt = get_checkpoints_version_dir()
    cfg = load_model_cfg(str(ckpt/"gen"), str(ckpt/"dis"))
    model, samplers = None, {}
    for k, g in boxes.items():
        info = make_sweep_volume_gripper_info(g["extents_open"], g["offset_open"], g["extents_mid"],
                                              g["offset_mid"], name=k.replace(" ", "_"))
        samplers[k] = GraspGenXSampler(cfg, gripper_info=info, model=model)
        model = samplers[k].model

    meta = {}
    for uid, name in OBJECTS.items():
        row = cat.loc[uid]
        assert row.split == "test" and row.short_mm < finray.TIP_GAP_OPEN, uid
        t = time.time()
        mesh = objects.objaverse(uid)
        pts = np.asarray(trimesh.sample.sample_surface(mesh, 3500, seed=0)[0], np.float32)
        centre = pts.mean(0)
        pts -= centre                                   # the model expects a centred cloud
        mesh.apply_translation(-centre)
        mesh.export(SIM/"objects"/f"{uid}.obj", include_texture=False)

        out = {"centre": centre}
        np.random.seed(0); torch.manual_seed(0)
        out["2f|poses"], out["2f|conf"] = diffusion(pts, samplers["tuned 2"])
        np.random.seed(0); torch.manual_seed(0)
        extra = diffusion(pts, samplers["stl 4"])[0]
        np.random.seed(0); torch.manual_seed(0)
        out["4f|poses"], out["4f|conf"] = four_finger.plan(pts, samplers["tuned 2"], topk=POOL,
                                                           extra_candidates=extra)[:2]
        for g, n in (("2f", 2), ("4f", 4)):
            out[f"{g}|poses"], out[f"{g}|conf"] = out[f"{g}|poses"][:TOP], out[f"{g}|conf"][:TOP]
            checks = [finray.Checker(mesh, n).check(P) for P in out[f"{g}|poses"]]
            out[f"{g}|valid"] = np.array([c["valid"] for c in checks])
            out[f"{g}|reason"] = np.array([c["reason"] for c in checks])
        np.savez_compressed(SIM/f"{uid}.npz", **out)
        meta[uid] = dict(name=name, label=row.label, up=row.up, long_mm=row.long_mm, mid_mm=row.mid_mm,
                         short_mm=row.short_mm, circularity=row.circularity)
        print(f"{name:16s} ({row.label})  2f valid {out['2f|valid'].sum()}/{TOP}  "
              f"4f valid {out['4f|valid'].sum()}/{TOP}  {time.time() - t:.1f}s", flush=True)
    (SIM/"objects.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
