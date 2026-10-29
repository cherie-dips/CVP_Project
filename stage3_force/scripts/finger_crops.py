"""A square crop around the marked finger in every frame, for the image models (force_vision.py).

The gripper is hand-held, so the finger moves, turns and changes size in the camera image. The crop
follows it using the six tracked markers (extract.py): centred a little past the middle of the marker
chain towards the finger's base, 2.4 chain lengths wide, and turned so the chain always runs from the
fingertip at the top to the base at the bottom. It takes in the whole finger -- contact beam, ribs,
back beam -- and the clamp it is mounted on, so a model can see the beam move relative to the rigid
gripper, which the six markers alone cannot show.

Frames where the markers were not found get the crop interpolated from the nearest frames that have
them; `gap` records how many frames away that is.

    python stage3_force/scripts/finger_crops.py
        -> data/forceSensor/vision/crops.npy    (frames, 224, 224, 3) uint8 RGB
           data/forceSensor/vision/index.csv    clip, frame, t, force_N, gap, chain_px, angle
           results/finger_crops.png             a contact sheet to check them
"""
import os, sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from crop import RESULTS, VIDEOS, gripper_video
from extract import N_MARKERS, OUT as FRAMES

VISION = os.path.join(VIDEOS, "vision")
SIZE = 224
WIDTH = 2.4          # crop side, in chain lengths
TOWARD_BASE = 0.2    # crop centre past the chain's middle, towards the base, in chain lengths
SMOOTH = 5           # frames, moving average on the crop parameters


def crop_params(d):
    """Per frame: chain middle (x, y), direction tip -> base (radians), chain length (px), and the
    distance in frames to the nearest frame where all six markers were found."""
    m = np.stack([d[[f"m{j}x", f"m{j}y"]].values for j in range(N_MARKERS)], 1)      # (T, 6, 2)
    found = d.markers_found.values.astype(bool)
    tip, base = m[:, 0].copy(), m[:, -1].copy()
    # the gripper is held from the bottom right in every clip, so the base is the end nearer that corner
    swap = tip.sum(1) > base.sum(1)
    tip[swap], base[swap] = base[swap], tip[swap]
    mid, vec = (tip + base) / 2, base - tip
    angle, length = np.arctan2(vec[:, 1], vec[:, 0]), np.hypot(*vec.T)

    i, idx = np.arange(len(d)), np.where(found)[0]
    if not len(idx):
        raise RuntimeError(f"clip {d['clip'].iloc[0]}: no frame with all six markers")
    ang = np.unwrap(angle[idx])
    P = np.stack([np.interp(i, idx, mid[idx, 0]), np.interp(i, idx, mid[idx, 1]),
                  np.interp(i, idx, ang), np.interp(i, idx, length[idx])], 1)
    k = np.ones(SMOOTH) / SMOOTH
    P = np.stack([np.convolve(np.pad(c, SMOOTH // 2, mode="edge"), k, "valid") for c in P.T], 1)
    gap = np.abs(i[:, None] - idx[None]).min(1)
    return P, gap


def warp(img, cx, cy, angle, length):
    """Crop turned so the tip -> base direction points down the image."""
    u = np.array([np.cos(angle), np.sin(angle)])
    c = np.array([cx, cy]) + TOWARD_BASE * length * u
    s = SIZE / (WIDTH * length)
    R = s * np.array([[u[1], -u[0]], [u[0], u[1]]])
    M = np.c_[R, SIZE / 2 - R @ c]
    return cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)


def main():
    df = pd.read_csv(FRAMES)
    os.makedirs(VISION, exist_ok=True)
    crops = np.lib.format.open_memmap(os.path.join(VISION, "crops.npy"), mode="w+", dtype=np.uint8,
                                      shape=(len(df), SIZE, SIZE, 3))
    rows, n = [], 0
    for k, d in df.groupby("clip", sort=True):
        P, gap = crop_params(d)
        cap = cv2.VideoCapture(gripper_video(k))
        for j in range(len(d)):
            ok, img = cap.read()
            if not ok:
                raise RuntimeError(f"clip {k}: video ended at frame {j}")
            crops[n] = warp(img, *P[j])[..., ::-1]
            n += 1
        rows.append(pd.DataFrame({"clip": k, "object": d.object.values, "frame": d.frame.values, "t": d.t.values,
                                  "force_N": d.force_N.values, "markers_found": d.markers_found.values,
                                  "gap": gap, "chain_px": P[:, 3].round(1), "angle": P[:, 2].round(3)}))
        print(f"clip {k}: {len(d)} frames, markers in {d.markers_found.mean():.0%}, "
              f"crop within 15 frames of a marker frame in {(gap <= 15).mean():.0%}")
    crops.flush()
    index = pd.concat(rows, ignore_index=True)
    index.to_csv(os.path.join(VISION, "index.csv"), index=False)

    # contact sheet: per clip, the frame of peak force and a frame without contact
    tiles = []
    for k, d in index.groupby("clip"):
        for j in (d.force_N.idxmax(), d.index[len(d) // 8]):
            t = crops[j].copy()
            cv2.putText(t, f"{k} {d.object[j][:12]} {d.force_N[j]:.1f}N", (4, 14), 0, 0.42, (255, 255, 255), 1)
            tiles.append(t)
    sheet = np.vstack([np.hstack(tiles[r * 10:(r + 1) * 10]) for r in range(len(tiles) // 10)])
    cv2.imwrite(os.path.join(RESULTS, "finger_crops.png"), sheet[..., ::-1])
    print(f"{n} crops -> {VISION}")


if __name__ == "__main__":
    main()
