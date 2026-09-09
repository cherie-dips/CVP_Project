#!/usr/bin/env python3
"""Ask a vision-language model which gripper configuration to use.

Our two grippers are the SAME fin-ray gripper: the 2-finger configuration is the 4-finger one with
two fingers removed. Finger stiffness, material and finger length are therefore identical, and the
only differences are contact count and how the contacts are arranged. That makes the question
unusually clean, and it is what the prompt tells the model.

Provider-agnostic. `prompts` writes ready-to-send requests, `score` grades answers against the
outcomes recorded in results/hardware_trials.csv, `table` renders the gripper/object table.

    python step2_gripper_vlm/vlm_gripper_select.py prompts --out prompts.jsonl
    # run them through any VLM, save {"object": {"gripper": "...", ...}} as answers.json
    python step2_gripper_vlm/vlm_gripper_select.py score --answers answers.json
    python step2_gripper_vlm/vlm_gripper_select.py table --answers answers.json
"""
import argparse, base64, json, os, pathlib, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
# The only vision model enabled on this NVIDIA account. Cosmos-Reason (NVIDIA's physical-reasoning
# VLM, the natural fit) and the 90B Llama both refuse or time out — see the README.
DEFAULT_MODEL = "meta/llama-3.2-11b-vision-instruct"

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
GRIPPER_DIR = REPO / "data/gripper"
TRIALS = HERE / "results/hardware_trials.csv"

# Measured on the bench (build step: press a finger onto a kitchen scale, carbon paper for area).
SPEC = {
    "2-finger": dict(contacts=2, aperture_mm=None, contact_area_cm2=None),
    "4-finger": dict(contacts=4, aperture_mm=None, contact_area_cm2=None),
}

SYSTEM = """You choose how a robot should pick up an object.

The robot has one 3D-printed fin-ray gripper that can be run in two configurations:

  2-FINGER: two opposing compliant fingers. A pinch from two sides.
  4-FINGER: the same gripper with two extra fingers fitted, closing from four sides. An enveloping
            grasp.

The fingers are IDENTICAL in both cases — same material, same length, same softness. Only the
number of contacts and their arrangement change. So do not reason about one being softer; reason
about how the contacts are arranged.

What follows from that:
- 4 contacts share the load, so each one presses with about half the force of 2 contacts. Better for
  FRAGILE objects, which fail from local pressure.
- 4 contacts surround the object, so it cannot slide out sideways. Better for ROUND or SLIPPERY
  objects.
- 4 contacts need room on all sides. The 4-finger CANNOT pick up something thin lying flat on a
  table, or reach into a narrow space. Use 2-finger there.
- 2 contacts suit objects with two flat parallel faces, and anything long and thin that can be
  pinched across its short axis.
- The gripper is driven by one motor and controlled by POSITION, not force. Closing too far is what
  breaks things.

Answer with JSON only, no other text:
{"gripper": "2-finger" or "4-finger",
 "confidence": number from 0 to 1,
 "risk": "crush" or "slip" or "cannot_reach" or "none",
 "reason": "under 15 words"}"""


def b64(path):
    return base64.b64encode(pathlib.Path(path).read_bytes()).decode()


def call_nvidia(key, model, system, user, img_b64, retries=3):
    """NVIDIA's vision endpoint takes the image as an <img> tag inside the message text."""
    content = f'{system}\n\n{user} <img src="data:image/jpeg;base64,{img_b64}" />'
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": 300, "temperature": 0}
    for i in range(retries):
        try:
            r = requests.post(NVIDIA_URL, headers={"Authorization": f"Bearer {key}"},
                              json=body, timeout=180)
            if r.status_code != 200:
                time.sleep(2 + 2 * i)
                continue
            txt = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", txt, re.S)
            return json.loads(m.group(0)) if m else {"raw": txt[:200]}
        except Exception:
            time.sleep(2 + 2 * i)
    return None


def cmd_run(args):
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        sys.exit("set NVIDIA_API_KEY")
    df = pd.read_csv(TRIALS)
    todo = [r for r in df.itertuples() if (GRIPPER_DIR / f"{r.object}.jpg").exists()]
    print(f"{args.model} on {len(todo)} objects")
    out = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(call_nvidia, key, args.model, SYSTEM,
                          f"Object: {r.object.replace('_',' ')}. Which configuration?",
                          b64(GRIPPER_DIR / f"{r.object}.jpg")): r.object for r in todo}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
    json.dump(out, open(args.out, "w"), indent=1)
    got = sum(1 for v in out.values() if v and "gripper" in v)
    print(f"parsed {got}/{len(todo)} replies -> {args.out}")
    for k, v in sorted(out.items()):
        if v and "gripper" in v:
            print(f"  {k:<14} {v['gripper']:<10} {str(v.get('reason',''))[:52]}")
        else:
            print(f"  {k:<14} FAILED  {str(v)[:60] if v else ''}")


def cmd_prompts(args):
    df = pd.read_csv(TRIALS)
    n = 0
    with open(args.out, "w") as fh:
        for r in df.itertuples():
            img = GRIPPER_DIR / f"{r.object}.jpg"
            if not img.exists():
                continue
            desc = r.description if isinstance(r.description, str) and r.description else r.object
            req = {
                "object": r.object,
                "system": SYSTEM,
                "user": f"Object: {desc.replace('_', ' ')}. Which configuration should the robot use?",
                "image_path": str(img),
            }
            if args.embed:
                req["image_b64"] = b64(img)
            fh.write(json.dumps(req) + "\n")
            n += 1
    print(f"wrote {n} prompts to {args.out}")
    print("The image shows the gripper already near the object, which is a hint the model gets for")
    print("free. Crop to the object alone if you want a clean test of object-only reasoning.")


def _load(args):
    ans = json.load(open(args.answers))
    df = pd.read_csv(TRIALS)
    df = df[df.gripper.notna() & df.outcome.notna()]
    if df.empty:
        sys.exit(f"No labelled trials yet — fill in `gripper` and `outcome` in {TRIALS}.\n"
                 f"Frame strips to help: data/gripper/frames/<object>_strip.jpg")
    return ans, df


def cmd_score(args):
    ans, df = _load(args)
    df = df[df.object.isin(ans)]
    if df.empty:
        sys.exit("No overlap between answers and labelled trials.")
    ok = held = 0
    print(f"{'object':<16}{'VLM says':<11}{'we used':<11}{'held?':<8} reason")
    for r in df.itertuples():
        a = ans[r.object]
        agree = a["gripper"] == r.gripper
        ok += agree
        held += str(r.outcome).lower().startswith("held")
        print(f"{r.object:<16}{a['gripper']:<11}{r.gripper:<11}{str(r.outcome):<8} "
              f"{'ok  ' if agree else 'diff'} {a.get('reason','')[:40]}")
    n = len(df)
    print(f"\nagrees with what we used: {ok}/{n} ({ok/n:.0%})")
    print(f"our trials that held:      {held}/{n}")
    print("\nNote: agreeing with our choice is not the same as being right. Only trials where we")
    print("ran BOTH configurations on the same object can say which one was actually better.")
    both = df.groupby("object").gripper.nunique()
    print(f"objects tried with both configurations: {(both > 1).sum()}")


def cmd_table(args):
    ans, df = _load(args)
    print("| Object | VLM picks | Reason | Risk | We used | Outcome |")
    print("|---|---|---|---|---|---|")
    for r in df.itertuples():
        a = ans.get(r.object, {})
        print(f"| {r.object} | {a.get('gripper','—')} | {a.get('reason','')} | "
              f"{a.get('risk','')} | {r.gripper} | {r.outcome} |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run"); rp.set_defaults(f=cmd_run)
    rp.add_argument("--model", default=DEFAULT_MODEL)
    rp.add_argument("--workers", type=int, default=4)
    rp.add_argument("--out", default=str(HERE / "results/vlm_answers.json"))
    p = sub.add_parser("prompts"); p.set_defaults(f=cmd_prompts)
    p.add_argument("--out", default=str(HERE / "results/vlm_prompts.jsonl"))
    p.add_argument("--embed", action="store_true", help="inline base64 images")
    for name, fn in (("score", cmd_score), ("table", cmd_table)):
        q = sub.add_parser(name); q.set_defaults(f=fn)
        q.add_argument("--answers", required=True)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
