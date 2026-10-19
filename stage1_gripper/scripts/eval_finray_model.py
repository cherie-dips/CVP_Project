"""Score Qwen3-VL on the held-out fin-ray gripper choice.

    python stage1_gripper/scripts/eval_finray_model.py <split.jsonl> <out.json> [adapter_path]

Reports accuracy and, because the test set carries the natural 1-in-5 label rate,
per-class recall -- a model that answers "2-finger" every time scores 96% overall
and 0% on the class that matters.
"""
import collections, json, os, pathlib, re, sys

from mlx_vlm import load, generate
from mlx_vlm.utils import load_config
from mlx_vlm.prompt_utils import apply_chat_template

MODEL = os.environ.get("VLM_MODEL", "mlx-community/Qwen3-VL-2B-Instruct-4bit")
split, out_file = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
adapter = sys.argv[3] if len(sys.argv) > 3 else None

rows = [json.loads(l) for l in open(split)]
model, processor = load(MODEL, adapter_path=adapter)
config = load_config(MODEL)

def parse(reply):
    lines = [l.strip() for l in reply.splitlines() if l.strip()]
    head = re.sub(r"^gripper:\s*", "", lines[0], flags=re.I) if lines else ""
    g = ("4-finger" if re.search(r"\b4[\s-]*finger", head, re.I) else
         "2-finger" if re.search(r"\b2[\s-]*finger", head, re.I) else None)
    m = re.search(r"reason:\s*(.+)", reply, re.I)
    return g, (m.group(1).strip()[:160] if m else "")

preds = []
for r in rows:
    prompt = apply_chat_template(processor, config, r["messages"][0]["content"], num_images=1)
    reply = generate(model, processor, prompt, image=r["images"],
                     max_tokens=80, temperature=0.0, verbose=False).text.strip()
    g, reason = parse(reply)
    preds.append(dict(uuid=r["uuid"], object=r["object"], truth=r["gripper"],
                      pred=g, reason=reason))
    print(f"{r['object'][:26]:26s} truth {r['gripper']:9s} pred {str(g):9s}", flush=True)

n = len(preds)
acc = sum(p["pred"] == p["truth"] for p in preds)/n
by_class = {}
for cls in ["2-finger", "4-finger"]:
    hits = [p for p in preds if p["truth"] == cls]
    by_class[cls] = dict(n=len(hits),
                         recall=round(sum(p["pred"] == cls for p in hits)/max(len(hits),1), 3))
dist = collections.Counter(p["pred"] for p in preds)

res = dict(model=MODEL, adapter=adapter, n=n, accuracy=round(acc, 3), by_class=by_class,
           predicted=dict(dist), preds=preds)
out_file.write_text(json.dumps(res, indent=1))
print(f"\naccuracy {acc:.1%}   "
      + "   ".join(f"{c} recall {v['recall']:.0%} (n={v['n']})" for c, v in by_class.items()))
print("predicted:", dict(dist))
