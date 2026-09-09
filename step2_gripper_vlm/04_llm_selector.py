#!/usr/bin/env python3
"""Ask an LLM which gripper to use, and score it against real grasp outcomes.

Runs on the SAME held-out test split as 02_train_selector.py, so the numbers are directly
comparable to the trained model and the baselines.

    export GROQ_API_KEY=...
    python step2_gripper_vlm/04_llm_selector.py --n 400
    python step2_gripper_vlm/04_llm_selector.py --n 400 --shots 8      # few-shot
"""
import argparse, json, os, pathlib, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import GroupShuffleSplit

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM = """You choose which of two robot grippers will grasp an object more reliably.

  A = NARROW parallel jaw, opens to 80 mm, slim fingers.
  B = WIDE parallel jaw, opens to 140 mm, broad fingers.

Guidance:
- Thin, flat or small objects suit A: B's fingers are too broad to close on them.
- Large, round or bulky objects suit B: A cannot open wide enough to get around them.
- Both are parallel jaws, so neither is gentler than the other.

Reply with JSON only: {"gripper":"A" or "B","confidence":0-1,"reason":"under 12 words"}"""


class TokenBucket:
    """Groq's free tier caps TOKENS per minute, not requests. Pace against that."""

    def __init__(self, tokens_per_min):
        self.rate = tokens_per_min / 60.0
        self.tokens = tokens_per_min * 0.5
        self.cap = tokens_per_min
        self.t = time.monotonic()
        self.lock = threading.Lock()

    def take(self, n):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.cap, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
                wait = (n - self.tokens) / self.rate
            time.sleep(min(wait, 5.0))


def ask(session, key, model, user, system, bucket, cost=280, retries=5):
    body = {"model": model, "temperature": 0, "max_tokens": 400, "reasoning_effort": "low",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    for attempt in range(retries):
        bucket.take(cost)
        try:
            r = session.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=90)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("retry-after", 5)) + attempt * 2)
                continue
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", txt, re.S)
            if m:
                return json.loads(m.group(0))
        except Exception:
            time.sleep(1 + attempt * 2)
    return None


def test_split(df):
    tv, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0)
                  .split(np.arange(len(df)), groups=df.category))
    return tv, te


def regret(choice, d):
    got = np.where(choice == "franka", d.franka_rate.values, d.robotiq_rate.values)
    return float((d.oracle_rate.values - got).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--n", type=int, default=400, help="test objects to score (0 = all)")
    ap.add_argument("--shots", type=int, default=0, help="few-shot examples from the train split")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tpm", type=int, default=7500, help="token/minute budget")
    a = ap.parse_args()
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        sys.exit("set GROQ_API_KEY")

    df = pd.read_csv(RES / "gripper_labels.csv")
    tv, te = test_split(df)
    rng = np.random.default_rng(0)
    if a.n and a.n < len(te):
        te = rng.choice(te, a.n, replace=False)
    dte = df.iloc[te]

    system = SYSTEM
    if a.shots:
        ex = df.iloc[rng.choice(tv, a.shots, replace=False)]
        lines = [f'  "{r.category.replace("_"," ")}" -> '
                 f'{"A" if r.gap > 0 else "B"}' for r in ex.itertuples()]
        system += "\n\nExamples from real grasp trials:\n" + "\n".join(lines)

    print(f"{a.model}   {len(dte)} objects   {a.shots}-shot")
    s = requests.Session()
    bucket = TokenBucket(a.tpm)
    est_min = len(dte) * 280 / a.tpm
    print(f"  rate limit {a.tpm} tokens/min -> about {est_min:.0f} min\n")
    out = {}
    with ThreadPoolExecutor(max_workers=a.workers) as exr:
        futs = {exr.submit(ask, s, key, a.model,
                           f"Object: {r.category.replace('_',' ')}.", system, bucket): r.Index
                for r in dte.itertuples()}
        done = 0
        for f in as_completed(futs):
            out[futs[f]] = f.result()
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(dte)}", flush=True)

    ok = [i for i in dte.index if out.get(i) and out[i].get("gripper") in ("A", "B")]
    d = df.loc[ok]
    choice = np.array(["franka" if out[i]["gripper"] == "A" else "robotiq" for i in ok])
    print(f"\nparsed {len(ok)}/{len(dte)} replies")

    res = {}
    def rep(name, ch):
        r = regret(ch, d); acc = float((ch == d.best.values).mean())
        got = np.where(ch == "franka", d.franka_rate.values, d.robotiq_rate.values).mean()
        res[name] = dict(regret=r, accuracy=acc, success=float(got))
        print(f"  {name:<34}{r:>9.4f}{acc:>10.3f}{got:>9.3f}")

    print(f"\n  {'strategy':<34}{'regret':>9}{'accuracy':>10}{'success':>9}")
    rep("always franka", np.full(len(d), "franka"))
    rep(f"LLM {a.shots}-shot", choice)
    rep("oracle", d.best.values)
    print(f"\n  LLM picked the wide gripper {100*(choice=='robotiq').mean():.0f}% of the time"
          f"   (it is actually better {100*(d.gap<0).mean():.0f}% of the time)")

    tag = f"{a.shots}shot"
    json.dump(dict(model=a.model, shots=a.shots, n=len(ok), results=res),
              open(RES / f"llm_selector_{tag}.json", "w"), indent=1)
    print(f"\nwrote {RES/f'llm_selector_{tag}.json'}")


if __name__ == "__main__":
    main()
