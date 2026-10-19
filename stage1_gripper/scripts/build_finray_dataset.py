"""Turn the 100 labelled Objaverse objects into an MLX-VLM LoRA dataset.

The split is deliberately balanced for training -- the natural label rate is
about 1 round object in 5, and training on that teaches the class prior, which
is the failure we are trying to fix. The test set keeps the natural rate so the
evaluation is not flattered.
"""
import json, pathlib, random

ROOT = pathlib.Path.cwd()
OUT  = ROOT/"stage1_gripper/results/labels"
DATA = OUT/"vlm_data"; DATA.mkdir(exist_ok=True)

PROMPT = """This object is called "{name}". The image shows two views of its 3D mesh.
It measures {short:.0f} mm across its shortest axis and {long:.0f} mm across its longest.

A robot must pick it up with one of two fin-ray grippers. The fingers are identical -- only how many close on the object differs:
  2-finger - two fingers close from opposite sides
  4-finger - four fingers close from four sides, caging the object

Which should it use? Answer in exactly two lines and nothing else:
gripper: 2-finger or 4-finger
reason: one short sentence"""

labels = json.loads((OUT/"labels_100.json").read_text())
four = [u for u, v in labels.items() if v["gripper"] == "4-finger"]
two  = [u for u, v in labels.items() if v["gripper"] == "2-finger"]
rng = random.Random(0); rng.shuffle(four); rng.shuffle(two)

n_tr, n_va = 13, 3
split = {
    "train": four[:n_tr]            + two[:n_tr],
    "valid": four[n_tr:n_tr+n_va]   + two[n_tr:n_tr+n_va],
    "test":  four[n_tr+n_va:]       + two[n_tr+n_va:],
}

for name, uuids in split.items():
    rows = []
    for u in uuids:
        v = labels[u]
        rows.append({
            "images": [str((OUT/"renders"/f"{u[:8]}.png").resolve())],
            "uuid": u, "object": v["name"], "gripper": v["gripper"],
            "messages": [
                {"role": "user",
                 "content": PROMPT.format(name=v["name"], short=v["short_mm"],
                                          long=v["long_mm"])},
                {"role": "assistant",
                 "content": f"gripper: {v['gripper']}\nreason: {v['reason']}"},
            ]})
    rng.shuffle(rows)
    (DATA/f"{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    n4 = sum(r["gripper"] == "4-finger" for r in rows)
    print(f"{name:6s} {len(rows):3d} examples  ({n4} 4-finger / {len(rows)-n4} 2-finger)")
