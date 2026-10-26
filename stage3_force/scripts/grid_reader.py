"""Read the numbers off the SEN0704 grid in the sensor crops (crop.py).

The Tactile Sensor Tool draws every number in one fixed font, centred in its
cell: digits 12 px tall on a fixed 9 px pitch. So a cell is read by trying
1-4 digit slots centred in the cell and matching each slot against a template
per digit. Compression that breaks a stroke or joins two digits does not
matter, because nothing is segmented.

The templates live in digit_templates.json next to this file. They were built
once from cells labelled by tesseract:

    python stage3_force/scripts/grid_reader.py      # rebuild + check digit_templates.json
"""
import json, os

import cv2
import numpy as np

TEMPLATES = os.path.join(os.path.dirname(__file__), "digit_templates.json")
TOP, DIGIT_H, PITCH = 28, 12, 9      # text line top row, digit height, digit pitch (px, in a cell crop)
PAD = 2                              # rows of vertical slack above and below the text line
INSET = (3, 2)                       # cell crop = grid lines + 3 px ... - 2 px, which drops the lines


def cells(sensor_frame, lay):
    """(row, col, cell image) for the 36 cells. The image is max(B, G, R): every
    cell colour is bright in some channel, and the black digits are dark in all."""
    B = sensor_frame.max(2)
    xs, ys = lay["cell_lines_x"], lay["cell_lines_y"]
    for r in range(6):
        for c in range(6):
            yield r, c, B[ys[r] + INSET[0]: ys[r + 1] - INSET[1], xs[c] + INSET[0]: xs[c + 1] - INSET[1]]


def ink(cell):
    """0 on the cell's background colour, 1 on black."""
    cell = cell.astype(np.float32)
    bg = np.median(cell)
    return np.clip((bg - cell) / max(bg, 1.0), 0, 1)


def load_templates(path=TEMPLATES):
    with open(path) as f:
        return np.array(json.load(f)["templates"], np.float32)


def read_number(cell, T, max_digits=4):
    """(value, cost): the best 1-4 digit reading of one cell, and the mean squared
    error per pixel inside its digit slots. Value is -1 for an empty cell.

    Ink outside the slots counts when choosing the reading (it is what tells "17"
    from "7"), but not in the returned cost, so a mouse cursor parked beside a
    number does not make an otherwise clean reading fail."""
    k = ink(cell)
    band = np.ascontiguousarray(k[TOP - PAD: TOP + DIGIT_H + PAD])
    if band.sum() < 3:
        return -1, np.inf
    W = band.shape[1]
    # squared error of every digit template at every (dy, x): (10, 2*PAD+1, W-PITCH+1)
    D = np.stack([cv2.matchTemplate(band, t, cv2.TM_SQDIFF) for t in T])
    # ink inside the text line, per column, for each dy -> what lies outside the slots costs its square
    sq = np.stack([(band[dy: dy + DIGIT_H] ** 2).sum(0) for dy in range(2 * PAD + 1)])
    cum = np.concatenate([np.zeros((sq.shape[0], 1), np.float32), np.cumsum(sq, 1)], 1)
    centre = (W - 1) / 2
    best = (np.inf, -1, np.inf)
    for n in range(1, max_digits + 1):
        x_first = int(round(centre + 0.5 - n * PITCH / 2))
        for dx in (-1, 0, 1):
            x0 = x_first + dx
            slots = [x0 + i * PITCH for i in range(n)]
            if x0 < 0 or slots[-1] + PITCH > W:
                continue
            for dy in range(2 * PAD + 1):
                cost = D[:, dy, slots]                      # (10, n)
                digits = cost.argmin(0)
                if n > 1 and digits[0] == 0:                # no leading zeros
                    digits[0] = 1 + cost[1:, 0].argmin()
                inside = cost[digits, np.arange(n)].sum()
                total = inside + cum[dy, x0] + (cum[dy, -1] - cum[dy, slots[-1] + PITCH])
                if total < best[0]:
                    best = (total, int("".join(map(str, digits))), inside / (DIGIT_H * PITCH * n))
    return best[1], best[2]


def read_grid(sensor_frame, lay, T, max_cost=0.02):
    """6x6 ints, -1 where a cell does not read cleanly (mouse cursor on top, blur)."""
    vals = np.full((6, 6), -1, int)
    for r, c, cell in cells(sensor_frame, lay):
        v, cost = read_number(cell, T)
        if cost <= max_cost:
            vals[r, c] = v
    return vals


# ── one-off: build the templates from tesseract labels, then check them ─────
def _tesseract(cell):
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        cv2.imwrite(f.name, cv2.resize(cell, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
        out = subprocess.run(["tesseract", f.name, "-", "--psm", "7", "-c",
                              "tessedit_char_whitelist=0123456789"], capture_output=True, text=True)
    txt = out.stdout.strip()
    return txt if txt.isdigit() else None


def _sample_cells(per_clip, seed, zeros_kept=0.05):
    from crop import CROPS, OBJECTS, sensor_video
    with open(os.path.join(CROPS, "layout.json")) as f:
        layouts = json.load(f)
    rng, out = np.random.default_rng(seed), []
    for k in sorted(OBJECTS):
        lay = layouts[str(k)]
        cap = cv2.VideoCapture(sensor_video(k))
        for t in rng.choice(lay["frames"], per_clip, replace=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
            _, f = cap.read()
            for r, c, cell in cells(f, lay):
                k_ = ink(cell)
                wide = (k_.max(0) > 0.3).sum() > 8           # more than one digit
                if wide or rng.random() < zeros_kept:
                    out.append((k, int(t), r, c, cell.copy()))
    return out


def build(per_clip=40, seed=0, rounds=3):
    from multiprocessing import Pool
    sample = _sample_cells(per_clip, seed)
    with Pool() as pool:
        labels = pool.map(_tesseract, [s[-1] for s in sample])
    data = [(ink(s[-1]), txt) for s, txt in zip(sample, labels) if txt and len(txt) <= 4]
    # round 0: slots placed by geometry alone; later rounds: shifted to best fit the templates
    T = None
    for _ in range(rounds):
        bank = [[] for _ in range(10)]
        for k, txt in data:
            band = k[TOP - PAD: TOP + DIGIT_H + PAD]
            n, W = len(txt), band.shape[1]
            x_first = int(round((W - 1) / 2 + 0.5 - n * PITCH / 2))
            shifts = [(dx, dy) for dx in (-1, 0, 1) for dy in range(2 * PAD + 1)] if T is not None else [(0, PAD)]
            def patches(dx, dy):
                return [band[dy: dy + DIGIT_H, x_first + dx + i * PITCH: x_first + dx + (i + 1) * PITCH]
                        for i in range(n)]
            if x_first - 1 < 0 or x_first + 1 + n * PITCH > W:
                continue
            dx, dy = min(shifts, key=lambda s: sum(((p - T[int(d)]) ** 2).sum()
                                                   for p, d in zip(patches(*s), txt)) if T is not None else 0)
            for p, d in zip(patches(dx, dy), txt):
                bank[int(d)].append(p)
        T = np.stack([np.median(b, 0) for b in bank]).astype(np.float32)   # median: robust to tesseract's slips
    with open(TEMPLATES, "w") as f:
        json.dump({"pitch": PITCH, "digit_height": DIGIT_H, "top": TOP,
                   "counts": [len(b) for b in bank],
                   "templates": np.round(T, 3).tolist()}, f)
    print("templates from", [len(b) for b in bank], "slots per digit ->", TEMPLATES)
    return T


def check(T, per_clip=25, seed=1):
    """Agreement with tesseract on cells not used to build the templates."""
    from multiprocessing import Pool
    sample = _sample_cells(per_clip, seed)
    with Pool() as pool:
        labels = pool.map(_tesseract, [s[-1] for s in sample])
    agree, differ, rejected = 0, [], []
    for (k, t, r, c, cell), txt in zip(sample, labels):
        if not txt:
            continue
        v, cost = read_number(cell, T)
        if cost > 0.02:
            rejected.append((k, t, r, c, txt, v, cost, cell))
        elif v == int(txt):
            agree += 1
        else:
            differ.append((k, t, r, c, txt, v, cost, cell))
    print(f"agree {agree}   differ {len(differ)}   rejected {len(rejected)}")
    return differ, rejected


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    T = build()
    differ, rejected = check(T)
    for k, t, r, c, txt, v, cost, _ in (differ + rejected)[:40]:
        print(f"  clip {k} frame {t} cell {r}{c}: tesseract {txt:>5}  ours {v:>5}  cost {cost:.4f}")
