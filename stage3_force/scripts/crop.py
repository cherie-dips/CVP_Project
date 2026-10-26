"""Split each screen recording in data/forceSensor/ into two videos:

    crops/clipNN_gripper.mp4   the gripper camera image (left of the recording)
    crops/clipNN_sensor.mp4    the 6x6 grid of DFRobot's Tactile Sensor Tool (SEN0704)

Both keep every frame of the original, so frame i of one lines up with frame i of
the other. The sensor crop is lossless (the digits are read from it); the gripper
crop is near-lossless. Crop boxes go to crops/layout.json, and a contact sheet of
every crop to results/crops.png.

    python stage3_force/scripts/crop.py
"""
import collections, itertools, json, os, subprocess

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIDEOS = os.path.join(ROOT, "data", "forceSensor")
CROPS = os.path.join(VIDEOS, "crops")
RESULTS = os.path.join(ROOT, "stage3_force", "results")

# What is grasped in each clip, numbered by the recording's file name. (1), a red bottle
# filmed from the side with the camera flat on the table, was dropped: every other clip
# looks down on the gripper.
OBJECTS = {
    2: "hexagon box",   3: "pink cube",    4: "orange cube",  5: "plush toy",
    6: "highlighter",   7: "highlighter",  8: "small red object",
    9: "pink box",     10: "pink box",    11: "jar lid",
}
MARGIN = 4          # px kept around the grid's outer lines


def source(k):
    return os.path.join(VIDEOS, f"WhatsApp Video 2026-10-26 at 00.06.30 ({k}).mp4")


def gripper_video(k):
    return os.path.join(CROPS, f"clip{k:02d}_gripper.mp4")


def sensor_video(k):
    return os.path.join(CROPS, f"clip{k:02d}_sensor.mp4")


# ── the grid: 7 evenly spaced dark lines each way ───────────────────────────
def _groups(a):
    return [int(round(np.mean(c))) for c in np.split(a, np.where(np.diff(a) > 3)[0] + 1)] if len(a) else []


def _seven_evenly_spaced(c):
    """The widest run of 7 equally spaced lines among the candidates."""
    c, best = np.array(c), None
    for i, j in itertools.combinations(range(len(c)), 2):
        d = c[j] - c[i]
        if d < 25:
            continue
        seq = [c[i] + k * d for k in range(7)]
        if all(np.abs(c - s).min() <= 3 for s in seq):
            seq = [int(c[np.abs(c - s).argmin()]) for s in seq]
            if best is None or d > best[1] - best[0]:
                best = seq
    return best


def _find_grid(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    bw = (g < 160).astype(np.uint8) * 255
    v = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(H * 0.25))))
    xs = _seven_evenly_spaced(_groups(np.where(v[:, W // 3:].sum(0) > 0)[0] + W // 3))
    if not xs:
        return None
    crop = bw[:, xs[0] + 2: xs[-1] - 2]
    h = cv2.morphologyEx(crop, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.8 * (xs[-1] - xs[0])), 1)))
    ys = _seven_evenly_spaced(_groups(np.where(h.sum(1) > 0)[0]))
    return (tuple(xs), tuple(ys)) if ys else None


def _even(v):
    return int(v) // 2 * 2


def detect_layout(path, samples=30):
    """Grid lines and camera rectangle, from frames spread over the whole clip.

    Raises if the grid is not found in the same place in every frame it is found in,
    i.e. if the GUI window moved during the recording.
    """
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    grids, grays = [], []
    for t in np.linspace(0, n - 1, samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, f = cap.read()
        if not ok:
            continue
        if (r := _find_grid(f)):
            grids.append(r)
        grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    counts = collections.Counter(grids)
    (xs, ys), hits = counts.most_common(1)[0]
    if len(counts) > 1:
        raise RuntimeError(f"{path}: the grid moved during the clip: {counts}")

    # Camera image: the largest region left of the GUI that is bright or changes over
    # time. The camera app around it is a constant dark grey.
    grays = np.array(grays)
    active = (grays.std(0) > 6) | (grays.mean(0) > 60)
    active[:, xs[0] - 230:] = False                         # the GUI's side panel and beyond
    active = cv2.morphologyEx(active.astype(np.uint8), cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    _, _, st, _ = cv2.connectedComponentsWithStats(active)
    x, y, w, h, _ = st[1 + st[1:, 4].argmax()]
    cam = [_even(x), _even(y), _even(w), _even(h)]

    gx, gy = _even(xs[0] - MARGIN), _even(ys[0] - MARGIN)
    grid = [gx, gy, _even(xs[-1] + MARGIN - gx + 2), _even(ys[-1] + MARGIN - gy + 2)]
    return {
        "frames": n,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "size": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
        "gripper_box": cam,                                   # x, y, w, h in the recording
        "sensor_box": grid,
        "cell_lines_x": [v - gx for v in xs],                 # grid lines inside the sensor crop
        "cell_lines_y": [v - gy for v in ys],
        "grid_found_in": f"{hits}/{samples} sampled frames",
    }


def write_crops(k, lay):
    (cx, cy, cw, ch), (sx, sy, sw, sh) = lay["gripper_box"], lay["sensor_box"]
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", source(k),
        "-filter_complex", f"[0:v]crop={cw}:{ch}:{cx}:{cy}[g];[0:v]crop={sw}:{sh}:{sx}:{sy}[s]",
        "-map", "[g]", "-c:v", "libx264", "-crf", "12", "-preset", "fast", "-fps_mode", "passthrough",
        "-pix_fmt", "yuv420p", gripper_video(k),
        "-map", "[s]", "-c:v", "libx264", "-crf", "0", "-preset", "fast", "-fps_mode", "passthrough",
        sensor_video(k),
    ], check=True)
    for p in (gripper_video(k), sensor_video(k)):             # same frames as the source, or stop
        n = int(cv2.VideoCapture(p).get(cv2.CAP_PROP_FRAME_COUNT))
        if n != lay["frames"]:
            raise RuntimeError(f"{p}: {n} frames, source has {lay['frames']}")


def contact_sheet(layouts, out):
    """Per clip: the recording with both boxes drawn, then the two crops, at the
    frame where the most grid cells are lit (i.e. mid-grasp)."""
    rows, H = [], 200
    for k, lay in layouts.items():
        cap = cv2.VideoCapture(source(k))
        sx, sy, sw, sh = lay["sensor_box"]
        best, frame = -1, None
        for t in np.linspace(0, lay["frames"] - 1, 40).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            _, f = cap.read()
            lit = (cv2.cvtColor(f[sy:sy + sh, sx:sx + sw], cv2.COLOR_BGR2HSV)[..., 1] > 80).mean()
            if lit > best:
                best, frame = lit, f
        full = frame.copy()
        for (x, y, w, h), col in ((lay["gripper_box"], (0, 200, 0)), (lay["sensor_box"], (255, 0, 255))):
            cv2.rectangle(full, (x, y), (x + w - 1, y + h - 1), col, 3)

        def fit(im):
            return cv2.resize(im, (int(im.shape[1] * H / im.shape[0]), H), interpolation=cv2.INTER_AREA)
        (cx, cy, cw, ch) = lay["gripper_box"]
        tiles = [fit(full), fit(frame[cy:cy + ch, cx:cx + cw]), fit(frame[sy:sy + sh, sx:sx + sw])]
        row = np.hstack([np.pad(t, ((4, 4), (4, 4), (0, 0)), constant_values=255) for t in tiles])
        cv2.putText(row, f"clip {k}: {OBJECTS[k]}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        rows.append(row)
    width = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows])
    cv2.imwrite(out, sheet)


def main():
    os.makedirs(CROPS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    layouts = {}
    for k in sorted(OBJECTS):
        lay = detect_layout(source(k))
        write_crops(k, lay)
        layouts[k] = lay
        print(f"clip {k:2d}  {lay['frames']:5d} frames  gripper {lay['gripper_box']}  "
              f"sensor {lay['sensor_box']}  grid found in {lay['grid_found_in']}", flush=True)
    with open(os.path.join(CROPS, "layout.json"), "w") as f:
        json.dump({str(k): v for k, v in layouts.items()}, f, indent=1)
    contact_sheet(layouts, os.path.join(RESULTS, "crops.png"))
    print("->", CROPS, "and", os.path.join(RESULTS, "crops.png"))


if __name__ == "__main__":
    main()
