"""Turn the cropped videos from crop.py into data, one row per frame.

    crops/clipNN_sensor.mp4   -> the 36 grid numbers (grid_reader.py) -> force in N (calibrate.py)
    crops/clipNN_gripper.mp4  -> the 6 yellow-green markers on the finger's contact beam

Outputs
    data/forceSensor/frames.csv       every frame of every clip
    data/forceSensor/qc/clipNN.mp4    the gripper crop with the markers found, beside the grid as read
    results/clip_summary.csv          one row per clip
    results/grasps.csv                one row per contact (force above CONTACT_N)
    results/clips.png                 per clip: force over time, marker coverage, the grasp at peak force

    python stage3_force/scripts/extract.py              # read every frame of every crop
    python stage3_force/scripts/extract.py --from-csv   # only rebuild the results/ files from frames.csv
"""
import itertools, json, os, sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import grid_reader
from calibrate import counts_to_newton
from crop import CROPS, OBJECTS, RESULTS, VIDEOS, gripper_video, sensor_video

OUT = os.path.join(VIDEOS, "frames.csv")
QC = os.path.join(VIDEOS, "qc")
N_MARKERS = 6
STRICT = ((22, 60, 140), (48, 255, 255))     # marker colour (HSV)
LOOSE = ((20, 30, 190), (50, 255, 255))      # overexposed markers: V clips at 255 and saturation drops
CONTACT_N = 0.5                              # below this the calibration is unreliable


# ── markers ─────────────────────────────────────────────────────────────────
def _blobs(frame, hsv_range, margin=6):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.morphologyEx(cv2.inRange(hsv, *hsv_range), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    H, W = m.shape
    return np.array([cen[i] for i in range(1, n)                   # markers are ~5-20 px across
                     if 4 <= st[i, 4] <= 400 and max(st[i, 2:4]) <= 3 * min(st[i, 2:4]) + 2
                     # the preview's rounded edge throws off coloured artefacts; a marker this close
                     # to the edge would be cut off anyway
                     and margin <= cen[i][0] < W - margin and margin <= cen[i][1] < H - margin]).reshape(-1, 2)


def _curve_error(pts):
    """Largest miss of the points from a quadratic, in the frame of their main axis (px)."""
    c = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    u, v = c @ vt[0], c @ vt[1]
    return np.abs(v - np.polyval(np.polyfit(u, v, 2), u)).max()


def _chain(pts):
    """The points ordered along the finger if they look like the marker chain, else None.
    In clean frames the markers miss a quadratic by ~2-3 px and the largest gap is at most
    1.43x the smallest."""
    if _curve_error(pts) > 5:
        return None
    c = pts - pts.mean(0)
    pts = pts[np.argsort(c @ np.linalg.svd(c, full_matrices=False)[2][0])]
    gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return pts if gaps.max() <= 1.6 * gaps.min() else None


def _best_chain(pts):
    """Of all 6-point subsets, the smoothest one that passes _chain, or None."""
    if not N_MARKERS <= len(pts) <= N_MARKERS + 4:
        return None
    subsets = sorted(itertools.combinations(range(len(pts)), N_MARKERS), key=lambda s: _curve_error(pts[list(s)]))
    for s in subsets:
        if (c := _chain(pts[list(s)])) is not None:
            return c
    return None


def find_markers(frame):
    """(6x2 marker positions ordered along the finger, how many came from the loose
    colour pass), or (None, 0) when the six cannot all be seen."""
    strict = _blobs(frame, STRICT)
    if (pts := _best_chain(strict)) is not None:
        return pts, 0
    loose = _blobs(frame, LOOSE)
    extra = np.array([p for p in loose if len(strict) == 0 or np.linalg.norm(strict - p, axis=1).min() > 4])
    need = N_MARKERS - len(strict)
    if 1 <= need <= 2 and need <= len(extra) <= 8:          # fill gaps with washed-out markers
        chains = [c for add in itertools.combinations(extra, need)
                  if (c := _chain(np.vstack([strict, np.array(add)]))) is not None]
        if chains:
            return min(chains, key=_curve_error), need
    if (pts := _best_chain(loose)) is not None:              # all of them washed out
        return pts, N_MARKERS
    return None, 0


# ── drawing, for the check videos ──────────────────────────────────────────
def _draw_grid(vals, force, size):
    img = np.full((size, size, 3), 255, np.uint8)
    cell = size // 6
    colours = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_VIRIDIS)[:, 0]
    for r in range(6):
        for c in range(6):
            v = vals[r, c]
            x0, y0 = c * cell, r * cell
            if v > 0:
                img[y0:y0 + cell, x0:x0 + cell] = colours[min(255, int(255 * v / 800))]
            cv2.rectangle(img, (x0, y0), (x0 + cell, y0 + cell), (180, 180, 180), 1)
            txt = str(v)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(img, txt, (x0 + (cell - tw) // 2, y0 + (cell + th) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255) if v > 400 else (0, 0, 0), 1, cv2.LINE_AA)
    return img


def _draw_frame(gripper, vals, force, pts, k, i, t):
    g = gripper.copy()
    if pts is not None:
        for j, (x, y) in enumerate(pts):
            cv2.circle(g, (int(round(x)), int(round(y))), 6, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(g, str(j), (int(x) + 7, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    h = g.shape[0]
    panel = np.hstack([g, _draw_grid(vals, force, h)])
    bar = np.full((28, panel.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, f"clip {k} {OBJECTS[k]}   t={t:6.2f}s  frame {i}   force {force:5.2f} N   "
                f"markers {'6/6' if pts is not None else '--'}", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return np.vstack([bar, panel])


# ── one clip ────────────────────────────────────────────────────────────────
def process(k):
    with open(os.path.join(CROPS, "layout.json")) as f:
        lay = json.load(f)[str(k)]
    T = grid_reader.load_templates()
    gcap, scap = cv2.VideoCapture(gripper_video(k)), cv2.VideoCapture(sensor_video(k))
    fps = lay["fps"]
    writer = None
    rows = []
    last_vals = np.zeros((6, 6), int)
    prev_cells, prev_read = [None] * 36, [None] * 36
    prev_pts, prev_i = None, -99
    for i in range(lay["frames"]):
        okg, g = gcap.read()
        oks, s = scap.read()
        if not (okg and oks):
            raise RuntimeError(f"clip {k}: stream ended at frame {i} of {lay['frames']}")
        # grid: re-read only the cells that changed since the last frame
        vals = np.full((6, 6), -1, int)
        for j, (r, c, cell) in enumerate(grid_reader.cells(s, lay)):
            if prev_cells[j] is not None and np.array_equal(cell, prev_cells[j]):
                v = prev_read[j]
            else:
                value, cost = grid_reader.read_number(cell, T)
                v = value if cost <= 0.06 else -1
                prev_cells[j], prev_read[j] = cell.copy(), v
            vals[r, c] = v
        unread = int((vals < 0).sum())
        vals = np.where(vals < 0, last_vals, vals)         # the GUI holds its last value; so do we
        last_vals = vals
        total = int(vals.sum())
        force = float(counts_to_newton(total))

        pts, filled = find_markers(g)
        if pts is not None and prev_pts is not None and i - prev_i <= 10:
            # keep marker 0 at the same end of the finger as in the last frame it was seen
            if np.linalg.norm(pts[::-1] - prev_pts) < np.linalg.norm(pts - prev_pts):
                pts = pts[::-1]
        if pts is not None:
            prev_pts, prev_i = pts, i

        rr, cc = np.mgrid[0:6, 0:6]
        row = {"clip": k, "object": OBJECTS[k], "frame": i, "t": round(i / fps, 4),
               "sum": total, "force_N": round(force, 4),
               "contact_cells": int((vals > 0).sum()),
               "centroid_row": round(float((rr * vals).sum() / total), 3) if total else np.nan,
               "centroid_col": round(float((cc * vals).sum() / total), 3) if total else np.nan,
               "unread_cells": unread,
               "markers_found": pts is not None, "markers_filled": filled}
        row.update({f"c{r}{c}": int(vals[r, c]) for r in range(6) for c in range(6)})
        for j in range(N_MARKERS):
            row[f"m{j}x"], row[f"m{j}y"] = (round(float(pts[j, 0]), 2), round(float(pts[j, 1]), 2)) \
                if pts is not None else (np.nan, np.nan)
        rows.append(row)

        frame = _draw_frame(g, vals, force, pts, k, i, i / fps)
        if writer is None:
            writer = cv2.VideoWriter(os.path.join(QC, f"clip{k:02d}.mp4"), cv2.VideoWriter_fourcc(*"avc1"),
                                     fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    writer.release()
    return pd.DataFrame(rows)


# ── per clip and per grasp summaries ────────────────────────────────────────
def grasps(df, min_frames=5, max_gap=10):
    """Runs of frames above CONTACT_N, joining runs split by short dips."""
    out = []
    for k, d in df.groupby("clip"):
        on = (d.force_N.values > CONTACT_N).astype(int)
        idx = np.where(on)[0]
        if not len(idx):
            continue
        runs = np.split(idx, np.where(np.diff(idx) > max_gap)[0] + 1)
        for run in runs:
            if len(run) < min_frames:
                continue
            seg = d.iloc[run[0]: run[-1] + 1]
            peak = seg.loc[seg.force_N.idxmax()]
            out.append({"clip": k, "object": OBJECTS[k],
                        "start_s": round(seg.t.iloc[0], 2), "end_s": round(seg.t.iloc[-1], 2),
                        "duration_s": round(seg.t.iloc[-1] - seg.t.iloc[0], 2),
                        "peak_N": round(peak.force_N, 2), "mean_N": round(seg.force_N.mean(), 2),
                        "peak_contact_cells": int(peak.contact_cells),
                        "peak_centroid_row": peak.centroid_row, "peak_centroid_col": peak.centroid_col,
                        "markers_found": round(seg.markers_found.mean(), 2)})
    return pd.DataFrame(out)


def clip_summary(df, g):
    rows = []
    for k, d in df.groupby("clip"):
        rows.append({"clip": k, "object": OBJECTS[k], "frames": len(d), "seconds": round(d.t.iloc[-1], 1),
                     "frames_in_contact": round((d.force_N > CONTACT_N).mean(), 3),
                     "peak_N": round(d.force_N.max(), 2),
                     "grasps": int((g["clip"] == k).sum()) if len(g) else 0,
                     "markers_found": round(d.markers_found.mean(), 3),
                     "markers_found_in_contact": round(d.markers_found[d.force_N > CONTACT_N].mean(), 3)
                     if (d.force_N > CONTACT_N).any() else np.nan,
                     "frames_with_unread_cells": round((d.unread_cells > 0).mean(), 4)})
    return pd.DataFrame(rows)


def plot_clips(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clips = sorted(df["clip"].unique())
    fig, axes = plt.subplots(len(clips), 3, figsize=(15, 2.3 * len(clips)),
                             gridspec_kw={"width_ratios": [4.2, 1, 1.4]})
    for ax, k in zip(axes, clips):
        d = df[df["clip"] == k]
        ax[0].plot(d.t, d.force_N, lw=0.9, color="C0")
        ymax = max(1.0, d.force_N.max() * 1.1)
        ax[0].fill_between(d.t, 0, ymax * 0.04, where=d.markers_found, color="C2", lw=0, step="mid",
                           label="all 6 markers seen")
        ax[0].axhline(CONTACT_N, color="0.6", lw=0.6, ls="--")
        ax[0].set_ylim(0, ymax); ax[0].set_xlim(d.t.min(), d.t.max())
        ax[0].set_ylabel("N"); ax[0].set_title(f"clip {k}: {OBJECTS[k]}", loc="left", fontsize=10)
        peak = d.loc[d.force_N.idxmax()]
        ax[0].axvline(peak.t, color="C3", lw=0.8)
        grid = np.array([[peak[f"c{r}{c}"] for c in range(6)] for r in range(6)])
        ax[1].imshow(grid, cmap="viridis", vmin=0, vmax=800)
        for r in range(6):
            for c in range(6):
                ax[1].text(c, r, int(grid[r, c]), ha="center", va="center", fontsize=6,
                           color="w" if grid[r, c] < 400 else "k")
        ax[1].set_title(f"peak {peak.force_N:.1f} N", fontsize=9); ax[1].axis("off")
        cap = cv2.VideoCapture(gripper_video(k))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(peak.frame))
        _, im = cap.read()
        if peak.markers_found:
            for j in range(N_MARKERS):
                cv2.circle(im, (int(peak[f"m{j}x"]), int(peak[f"m{j}y"])), 7, (255, 0, 255), 2)
        ax[2].imshow(im[..., ::-1]); ax[2].axis("off"); ax[2].set_title(f"t = {peak.t:.1f} s", fontsize=9)
    axes[0][0].legend(loc="upper right", fontsize=8)
    axes[-1][0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main(from_csv=False):
    """from_csv: skip the videos and rebuild the summaries from the existing frames.csv,
    keeping only the clips in OBJECTS."""
    clips = sorted(OBJECTS)
    if from_csv:
        df = pd.read_csv(OUT)
        df = df[df["clip"].isin(clips)].reset_index(drop=True)
    else:
        os.makedirs(QC, exist_ok=True)
        order = sorted(clips, key=lambda k: -json.load(open(os.path.join(CROPS, "layout.json")))[str(k)]["frames"])
        with Pool(min(len(clips), os.cpu_count())) as pool:
            parts = dict(zip(order, pool.map(process, order)))
        df = pd.concat([parts[k] for k in clips], ignore_index=True)
    df.to_csv(OUT, index=False)
    g = grasps(df)
    g.to_csv(os.path.join(RESULTS, "grasps.csv"), index=False)
    s = clip_summary(df, g)
    s.to_csv(os.path.join(RESULTS, "clip_summary.csv"), index=False)
    plot_clips(df, os.path.join(RESULTS, "clips.png"))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(s.to_string(index=False))
    print(f"\n{len(df)} frames -> {OUT}\n{len(g)} grasps -> results/grasps.csv\ncheck videos -> {QC}")


if __name__ == "__main__":
    main(from_csv="--from-csv" in sys.argv)
