#!/usr/bin/env python3
"""Step 2c. Turn the grasp videos into structured, labellable data.

For each grasp video we pull out a motion profile and four key frames — approach, the moment the
fingers close, the lift, and the end — plus a contact strip for fast labelling.

Uses ffmpeg + PIL only. (Installing OpenCV pulls numpy 2.x, which breaks graspgenx's pinned 1.26.4.)

    python step2_gripper_vlm/05_extract_videos.py
"""
import argparse, csv, json, pathlib, subprocess, sys
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
SOURCES = [REPO / "data/gripper", pathlib.Path.home() / "Documents/SP Dutt Video"]
OUT = REPO / "data/grasp_clips"
W, H = 160, 120          # size used for the motion signal


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True).stdout
    j = json.loads(out or "{}")
    st = (j.get("streams") or [{}])[0]
    num, _, den = (st.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return dict(width=st.get("width"), height=st.get("height"), fps=round(fps, 2),
                seconds=round(float(j.get("format", {}).get("duration", 0) or 0), 2))


def gray_stack(path, fps=10):
    """Decode the whole clip small and grey, as one array."""
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={W}:{H},format=gray",
         "-f", "rawvideo", "-"], capture_output=True)
    buf = np.frombuffer(p.stdout, np.uint8)
    n = len(buf) // (W * H)
    return buf[: n * W * H].reshape(n, H, W).astype(np.float32) if n else None


def analyse(frames):
    """Motion per frame, and the moments that matter."""
    d = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    if len(d) < 4:
        return None
    d_s = np.convolve(d, np.ones(3) / 3, mode="same")
    peak = int(np.argmax(d_s))                       # fingers closing / lifting: biggest change
    quiet = d_s < (d_s.mean() * 0.5)
    settle = int(np.argmax(quiet[peak:]) + peak) if quiet[peak:].any() else len(d_s) - 1
    # how much the scene changed from start to end, in the lower half where the object sat
    table = np.abs(frames[-1, H // 2:] - frames[0, H // 2:]).mean()
    return dict(motion=d_s, peak=peak, settle=settle,
                table_change=float(table), motion_peak=float(d_s[peak]),
                motion_end=float(d_s[-5:].mean()))


def save_frames(path, idx, fps, out_dir, stem):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for tag, i in idx.items():
        t = i / fps
        f = out_dir / f"{stem}_{tag}.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(path),
                        "-frames:v", "1", "-vf", "scale=320:-1", str(f)], check=False)
        paths.append(f)
    strip = out_dir / f"{stem}_strip.jpg"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *sum([["-i", str(p)] for p in paths], []),
                    "-filter_complex", f"hstack=inputs={len(paths)}", str(strip)], check=False)
    return strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "results/video_manifest.csv"))
    ap.add_argument("--out", default=str(HERE / "results/grasp_clips.csv"))
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()

    keep = set()
    mf = pathlib.Path(a.manifest)
    if mf.exists():
        for r in csv.DictReader(open(mf)):
            if r.get("kind") == "grasp":
                keep.add(r["file"])

    vids = []
    for src in SOURCES:
        if not src.exists():
            continue
        for p in sorted(src.iterdir()):
            if p.suffix.lower() not in (".mov", ".mp4"):
                continue
            if src.name == "SP Dutt Video" and p.name not in keep:
                continue
            vids.append(p)
    print(f"{len(vids)} grasp clips\n")

    rows = []
    for p in vids:
        meta = probe(p)
        fr = gray_stack(p, a.fps)
        if fr is None or len(fr) < 5:
            print(f"  {p.name}: could not decode")
            continue
        an = analyse(fr)
        if an is None:
            continue
        stem = p.stem
        idx = {"start": 0, "close": an["peak"],
               "lift": min(len(fr) - 1, an["settle"]), "end": len(fr) - 1}
        strip = save_frames(p, idx, a.fps, OUT, stem)
        rows.append(dict(clip=stem, source=p.parent.name, **meta,
                         close_s=round(an["peak"] / a.fps, 2),
                         settle_s=round(an["settle"] / a.fps, 2),
                         motion_peak=round(an["motion_peak"], 2),
                         motion_end=round(an["motion_end"], 2),
                         table_change=round(an["table_change"], 2),
                         strip=str(strip.relative_to(REPO)),
                         object="", gripper="", outcome=""))
        print(f"  {stem:<16} {meta['seconds']:>5.1f}s  close@{idx['close']/a.fps:>4.1f}s  "
              f"table change {an['table_change']:>5.1f}")

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} clips -> {a.out}")
    print(f"frames and strips -> {OUT}")
    print("\nFill in `object`, `gripper` (2-finger/4-finger) and `outcome` (held/slipped/dropped).")
    print("The strip shows start | close | lift | end, which is usually enough to read off both.")


if __name__ == "__main__":
    main()
