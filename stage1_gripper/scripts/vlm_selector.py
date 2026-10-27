"""The gripper selector as a classifier: the VLM's "2" against its "4", read from the logits.

The first fine-tune generated "gripper: X\\nreason: ..." and parsed the text, so all it could
report was the label greedy decoding happened to produce. Here the decision is read straight from
the model:

    score = logit("4") - logit("2")     at the first answer token

That ranks objects even when greedy decoding always says the same thing (it is an AUC away from
a decision), and the threshold is set on the validation split rather than fixed at 0, which
corrects a model that leans one way for every object (Zhao et al. 2021, "Calibrate before use").

Training is LoRA on the language model, with the loss restricted to those two tokens and weighted
by inverse class frequency, so the 13% of objects that need four fingers carry half the loss. Nothing else is in the
target: no reason text, so no number to make up. Each epoch is scored on the validation split and
the best one is kept.

    python vlm_selector.py score --layout axis [--adapter DIR] [--model M] [--tag NAME] [--splits ...]
        one forward pass per object -> results/selector/scores_<tag>.csv, plus the hidden state at
        the answer position -> data/selector/features_<tag>.npz (for the linear probe)
    python vlm_selector.py train --layout axis --out DIR [--epochs 3] [--model M]

Runs in the mlxvlm env (MLX and GraspGenX cannot share one).
"""
import argparse, json, math, pathlib, random, time

import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
import numpy as np, pandas as pd
from mlx.utils import tree_map
from PIL import Image, ImageOps
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model, save_adapter
from mlx_vlm.utils import load_config, prepare_inputs

mx.set_cache_limit(1 << 30)      # MLX otherwise keeps every freed buffer; 16 GB is shared with GraspGenX

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT/"stage1_gripper"/"results"/"selector"
IMAGES = ROOT/"data"/"selector"
QWEN = "mlx-community/Qwen3-VL-2B-Instruct-4bit"

GRIPPER = ("A soft robotic gripper picks it up with either 2 fingers, closing from opposite sides, "
           "or 4 fingers, closing from four sides and caging it. Two fingers need flat sides to pinch; "
           "an object whose cross-section is round would roll out of a two-finger pinch, so it gets four.\n\n"
           "Which does this object need? Answer with exactly one of: 2-finger, 4-finger.")
PROMPTS = {
    "axis": ("The image shows one object three times, each view scaled to fit its panel: on the left "
             "looking straight down its long axis, so you see its cross-section; in the middle and on "
             "the right, from two sides.\n\n" + GRIPPER),
    "fixed": "The image shows one object from two directions.\n\n" + GRIPPER,
}
PANELS = {"axis": 3, "fixed": 2}


class Selector:
    def __init__(self, model_path=QWEN, layout="axis", adapter=None, lora=None):
        self.model, self.processor = load(model_path, adapter_path=adapter)
        if lora:
            self.model = get_peft_model(self.model, find_all_linear_names(self.model.language_model),
                                        rank=lora["rank"], alpha=lora["alpha"], dropout=lora["dropout"],
                                        verbose=True)
        self.layout = layout
        config = load_config(model_path)
        prompt = apply_chat_template(self.processor, config, PROMPTS[layout], num_images=1)
        # The two answers share no prefix in Qwen ("2"/"4" are single tokens); SmolVLM puts a space
        # token first. Teacher-force whatever they share, compare the first token that differs.
        tok = self.processor.tokenizer
        lead = "" if prompt.endswith("\n") else " "
        a, b = (tok.encode(prompt + lead + s, add_special_tokens=False) for s in ("2-finger", "4-finger"))
        n = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        self.prompt = prompt + lead
        self.options = mx.array([a[n], b[n]])                   # [2-finger, 4-finger]

    def inputs(self, image):
        x = prepare_inputs(self.processor, images=[image], prompts=[self.prompt])
        rest = {k: v for k, v in x.items() if k not in ("input_ids", "pixel_values", "attention_mask")}
        return x["input_ids"], x["pixel_values"], x["attention_mask"], rest

    def forward(self, image, hidden=False):
        ids, pix, mask, rest = self.inputs(image)
        out = self.model(ids, pix, mask, **rest)
        two = out.logits[0, -1].astype(mx.float32)[self.options]
        return (two, _CAPTURED["h"][0, -1]) if hidden else two

    def capture_hidden(self):
        """Keep the language model's last hidden state (what goes into the LM head) on every call,
        for the linear probe. The outputs carry no hidden states, so wrap whatever produces it:
        the decoder in Qwen3-VL, the final norm in SmolVLM2."""
        lm = self.model.language_model
        if "model" in lm:
            inner = type(lm["model"])
            if getattr(inner, "_captures", False):
                return
            call = inner.__call__

            def wrapped(module, *a, **k):
                h = call(module, *a, **k)
                _CAPTURED["h"] = h
                return h
            inner.__call__, inner._captures = wrapped, True
        else:
            lm.norm = _Capture(lm.norm)


_CAPTURED = {}


class _Capture(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    @property
    def weight(self):                       # the decoder reads norm.weight.dtype
        return self.inner.weight

    def __call__(self, x):
        _CAPTURED["h"] = y = self.inner(x)
        return y


def image_path(layout, uid):
    return str(IMAGES/layout/f"{uid}.png")


def augment(path, layout, rng):
    """Mirror each panel at random: the cross-section's roundness does not change under reflection."""
    im = Image.open(path).convert("RGB")
    n, gap = PANELS[layout], 4
    w = (im.width - (n - 1) * gap) // n
    for i in range(n):
        box = (i * (w + gap), 0, i * (w + gap) + w, im.height)
        p = im.crop(box)
        if rng.random() < 0.5:
            p = ImageOps.mirror(p)
        if rng.random() < 0.5:
            p = ImageOps.flip(p)
        im.paste(p, box[:2])
    return im


# ── metrics, on the validation split only ───────────────────────────────────
def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    order = np.argsort(s)
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):                                   # average ranks over ties
        r[s == v] = r[s == v].mean()
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def best_threshold(y, s):
    """The threshold that maximises balanced accuracy on (y, s)."""
    y, s = np.asarray(y), np.asarray(s)
    cands = np.unique(s)
    cands = np.concatenate([[cands[0] - 1], (cands[1:] + cands[:-1]) / 2, [cands[-1] + 1]])
    ba = [((s[y == 1] > t).mean() + (s[y == 0] <= t).mean()) / 2 for t in cands]
    i = int(np.argmax(ba))
    return float(cands[i]), float(ba[i])


# ── commands ────────────────────────────────────────────────────────────────
def score(args):
    df = pd.read_csv(OUT/"objects.csv")
    if args.splits:
        df = df[df.split.isin(args.splits.split(","))].reset_index(drop=True)
    sel = Selector(args.model, args.layout, adapter=args.adapter)
    sel.model.eval()
    sel.capture_hidden()
    tag = args.tag or f"{args.layout}_{'lora' if args.adapter else 'zeroshot'}"
    rows, feats = [], []
    t0 = time.time()
    for i, r in enumerate(df.itertuples(), 1):
        two, h = sel.forward(image_path(args.layout, r.id), hidden=True)
        mx.eval(two)
        rows.append(dict(id=r.id, split=r.split, label=r.label, logit_2=float(two[0]), logit_4=float(two[1]),
                         score=float(two[1] - two[0])))
        feats.append(np.array(h.astype(mx.float32)))
        if i % 100 == 0:
            print(f"  {i}/{len(df)}  {(time.time() - t0) / i:.2f} s/object", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT/f"scores_{tag}.csv", index=False)
    if feats:
        np.savez_compressed(IMAGES/f"features_{tag}.npz", ids=out.id.values, X=np.stack(feats))
    v = out[out.split == "valid"]
    y = (v.label == "4-finger").astype(int)
    print(f"{tag}: valid AUC {auc(y, v.score):.3f}, says 4-finger for {(out.score > 0).mean():.0%} of objects at threshold 0")


def train(args):
    df = pd.read_csv(OUT/"objects.csv")
    tr, va = df[df.split == "train"].reset_index(drop=True), df[df.split == "valid"].reset_index(drop=True)
    lora = dict(rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    sel = Selector(args.model, args.layout, lora=lora)
    model = sel.model

    y_tr = (tr.label == "4-finger").astype(int).values
    w = np.where(y_tr == 1, len(y_tr) / (2 * y_tr.sum()), len(y_tr) / (2 * (len(y_tr) - y_tr.sum())))
    print(f"train {len(tr)} ({y_tr.sum()} 4-finger), valid {len(va)}; class weights "
          f"2-finger {w[y_tr == 0][0]:.2f}, 4-finger {w[y_tr == 1][0]:.2f}")

    steps = args.epochs * math.ceil(len(tr) / args.accum)
    warm = max(1, int(0.05 * steps))
    sched = optim.join_schedules([optim.linear_schedule(1e-7, args.lr, warm),
                                  optim.cosine_decay(args.lr, steps - warm, 1e-6)], [warm])
    opt = optim.Adam(learning_rate=sched)

    def loss_fn(model, image, y, weight):
        ids, pix, mask, rest = sel.inputs(image)
        out = model(ids, pix, mask, **rest)
        two = out.logits[:, -1, :].astype(mx.float32)[:, sel.options]
        return weight * nn.losses.cross_entropy(two, y).mean()

    vg = nn.value_and_grad(model, loss_fn)
    rng = random.Random(0)
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    log, best = [], (-1.0, None)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(tr))); rng.shuffle(order)
        acc, running, t0 = None, [], time.time()
        for k, i in enumerate(order, 1):
            loss, g = vg(model, augment(image_path(args.layout, tr.id[i]), args.layout, rng),
                         mx.array([int(y_tr[i])]), float(w[i]))
            acc = g if acc is None else tree_map(lambda a, b: a + b, acc, g)
            mx.eval(acc, loss)                     # MLX is lazy: without this every step's graph is kept
            running.append(loss.item())
            if k % args.accum == 0 or k == len(order):
                opt.update(model, tree_map(lambda a: a / args.accum, acc))
                mx.eval(model.parameters(), opt.state)
                acc = None
            if k % 100 == 0:
                print(f"  epoch {epoch}  {k}/{len(order)}  loss {np.mean(running[-100:]):.3f}  "
                      f"{(time.time() - t0) / k:.2f} s/step", flush=True)

        model.eval()
        s = []
        for r in va.itertuples():
            two = sel.forward(image_path(args.layout, r.id)); mx.eval(two)
            s.append(float(two[1] - two[0]))
        y = (va.label == "4-finger").astype(int).values
        thr, ba = best_threshold(y, s)
        a = auc(y, s)
        log.append(dict(epoch=epoch, train_loss=float(np.mean(running)), valid_auc=a, valid_bal_acc=ba,
                        threshold=thr, minutes=(time.time() - t0) / 60))
        print(f"epoch {epoch}: train loss {np.mean(running):.3f}  valid AUC {a:.3f}  "
              f"balanced acc {ba:.3f} at threshold {thr:+.2f}", flush=True)
        if a > best[0]:
            best = (a, epoch)
            save_adapter(model, out_dir/"adapters.safetensors")
    pd.DataFrame(log).to_csv(out_dir/"training_log.csv", index=False)
    (out_dir/"selector.json").write_text(json.dumps(
        {"model": args.model, "layout": args.layout, "best_epoch": best[1], "lora": lora,
         "lr": args.lr, "epochs": args.epochs, "grad_accum": args.accum,
         "class_weights": {"2-finger": float(w[y_tr == 0][0]), "4-finger": float(w[y_tr == 1][0])}}, indent=1))
    print(f"kept epoch {best[1]} (valid AUC {best[0]:.3f}) -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "train"])
    ap.add_argument("--layout", default="axis", choices=list(PROMPTS))
    ap.add_argument("--model", default=QWEN)
    ap.add_argument("--adapter")
    ap.add_argument("--tag")
    ap.add_argument("--splits", help="comma-separated, e.g. valid,test,external (default: all)")
    ap.add_argument("--out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    a = ap.parse_args()
    score(a) if a.cmd == "score" else train(a)
