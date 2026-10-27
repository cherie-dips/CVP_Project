"""The whole pipeline on the 50 real-scanned objects: stage 1 picks the gripper, stage 2 plans it.

No new inference: it joins stage 1's selector scores (vlm_selector.py) with stage 2's checked grasps
(evaluate_grasps.py). An object counts as handled when the grasp the pipeline would execute is
valid with the real fingers -- the planner's top-scored grasp ("top-1"), or, with finray.Checker
used as a filter, the best-scored valid grasp among the 100 highest ("filtered").

    /opt/anaconda3/envs/graspgenx/bin/python stage2_grasp/scripts/pipeline_check.py [selector run]
        -> stage2_grasp/results/pipeline_check.csv
"""
import pathlib, sys

import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
SEL = ROOT/"stage1_gripper"/"results"/"selector"
sys.path.insert(0, str(ROOT/"stage1_gripper"/"scripts"))
from selector_report import best_threshold, probe_scores

PLANNER = {  # (old pipeline, fixed pipeline) per configuration; the box tuning kept the old 2-finger box
    "2-finger": ("2f legacy", "2f legacy"),
    "4-finger": ("4f legacy", "4f composed"),
}


def selector_choice(run):
    """The selector's pick for every object, at the threshold set on the validation split."""
    objects = pd.read_csv(SEL/"objects.csv")
    if run.endswith("_probe"):
        s = probe_scores(run[:-len("_probe")], objects)
    else:
        s = pd.read_csv(SEL/f"scores_{run}.csv")
    s = s.set_index("id")
    va = s[s.split == "valid"]
    thr = best_threshold((va.label == "4-finger").astype(int).values, va.score.values)
    return {i: ("4-finger" if v > thr else "2-finger") for i, v in s.score.items()}


def main(run):
    grasps = pd.read_csv(ROOT/"stage2_grasp"/"results"/"grasp_check.csv").set_index(["id", "method"])
    objs = grasps.reset_index()[["id", "object", "rule"]].drop_duplicates().set_index("id")
    choice = selector_choice(run)
    policies = {
        "old: always 2-finger (what the VLM answered)": ({i: "2-finger" for i in objs.index}, 0),
        "old planners, rule picks": (dict(objs.rule), 0),
        "always 2-finger, fixed planner": ({i: "2-finger" for i in objs.index}, 1),
        "always 4-finger, fixed planner": ({i: "4-finger" for i in objs.index}, 1),
        f"selector ({run}) picks, fixed planners": (choice, 1),
        "rule picks, fixed planners": (dict(objs.rule), 1),
    }
    rows = []
    for name, (pick, fixed) in policies.items():
        top1, filt = [], []
        for i in objs.index:
            g = grasps.loc[(i, PLANNER[pick[i]][fixed])]
            top1.append(bool(g.top1_valid)); filt.append(bool(g.any_valid))
        rows.append(dict(policy=name, objects=len(objs), top1_valid=np.mean(top1), filtered_valid=np.mean(filt),
                         picks_4finger=np.mean([pick[i] == "4-finger" for i in objs.index])))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT/"stage2_grasp"/"results"/"pipeline_check.csv", index=False, float_format="%.3f")
    with pd.option_context("display.width", 160, "display.float_format", "{:.2f}".format):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "axis_lora")
