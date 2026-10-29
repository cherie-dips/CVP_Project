"""Videos of the PyBullet runs (sim_grasps.py): each object's #1 grasp with the 4-finger and the 2-finger
gripper, rack model, the whole sequence (approach, close, settle, lift, hold, shake), at real time.

It reruns sim_grasps.Scene.trial on a fresh scene, exactly as sim_grasps.py ran each planner's #1 grasp,
and grabs a TinyRenderer frame every 1/30 s of simulated time; rendering does not touch the physics, so
each video's outcome is checked against results/sim_grasps.csv. One camera per object, shared by both
grippers: from the side, 20 deg above, the lift pointing up on screen, framing the object and both
approaches, following the gripper once it lifts.

    /opt/anaconda3/envs/graspgenx/bin/python stage2_grasp/scripts/sim_videos.py
        -> stage2_grasp/results/sim_videos/<object>.mp4  left 4-finger, right 2-finger
           stage2_grasp/results/sim_videos/all.mp4       all 16 runs, two objects per row
"""
import json, pathlib, shutil, subprocess, sys, tempfile

import numpy as np, pandas as pd, pybullet as p
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sim_grasps as sg

OUT = sg.OUT/"sim_videos"
FPS, W, H = 30, 480, 360
SMALL = (320, 240)                                     # a panel in the overview grid
END_HOLD = 1.0                                         # s the final state stays on screen with the outcome
FOV = 45.0
FONT = fm.findfont("DejaVu Sans")
FONT_BOLD = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))
GREEN, RED = (30, 123, 52), (190, 30, 30)


def camera(scene, poses):
    """One camera for both grippers on this object: (target, eye direction, distance)."""
    up = scene.up
    hull = sg.trimesh.load(sg.SIM/"objects"/f"{scene.uid}.obj", force="mesh").bounding_box.vertices
    box = np.array([[x, y, z] for x in (-.08, .08) for y in (-.08, .08) for z in (0, .15)])  # open gripper
    pts = [hull]
    for P in poses:
        for back in (sg.BACKOFF, 0.0):
            T = P[:3, 3] - back * P[:3, 2]
            pts.append(box @ P[:3, :3].T + T)
    pts = np.vstack(pts)
    horiz = lambda v: v - (v @ up) * up
    best = None
    for az in np.radians(np.arange(0, 360, 15)):
        a0 = horiz(np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 0, 1.0]))
        a0 /= np.linalg.norm(a0)
        d = np.cos(az) * a0 + np.sin(az) * np.cross(up, a0)
        score = 0.0
        for P in poses:
            ah = horiz(P[:3, 2])
            if np.linalg.norm(ah) > 1e-6:                  # see the approach from the side
                score += np.linalg.norm(ah) * (1 - abs(d @ ah / np.linalg.norm(ah)))
            g = horiz(P[:3, 3] - scene.com)
            if np.linalg.norm(g) > 1e-6:                   # and the gripper in front of the object
                score += 0.4 * d @ g / np.linalg.norm(g)
        if best is None or score > best[0]:
            best = (score, d)
    d = best[1]
    view = np.cos(np.radians(20)) * d + np.sin(np.radians(20)) * up
    right = np.cross(up, view); right /= np.linalg.norm(right)
    upv = np.cross(view, right)
    c = (pts.min(0) + pts.max(0)) / 2
    w, h = np.ptp((pts - c) @ right), np.ptp((pts - c) @ upv)
    t = np.tan(np.radians(FOV / 2))
    dist = 1.25 * max(h / (2 * t), w / (2 * t * W / H))
    c = c + 0.07 * (2 * dist * t) * upv                    # keep the scene clear of the labels at the top
    return c, view, dist


class Recorder:
    """Called after every physics step; keeps a frame every 1/30 s of simulated time."""

    def __init__(self, cam, up):
        self.target0, self.view, self.dist = cam
        self.up = up
        self.every = int(round(1 / FPS / sg.DT))
        self.k, self.frames, self.phases = 0, [], []
        self.proj = p.computeProjectionMatrixFOV(fov=FOV, aspect=W / H, nearVal=0.02, farVal=5.0)

    def __call__(self, scene):
        self.k += 1
        if self.k % self.every == 0:
            self.frames.append(self.render(scene))
            self.phases.append(scene.phase)

    def render(self, scene):
        target = self.target0
        if scene.phase in ("lift", "hold", "shake"):       # follow the gripper up and through the shake
            target = target + (sg.pose(scene.grip)[0] - scene.goal)
        eye = target + self.dist * self.view
        vm = p.computeViewMatrix(list(eye), list(target), list(self.up))
        img = p.getCameraImage(W, H, vm, self.proj, renderer=p.ER_TINY_RENDERER, shadow=0,
                               lightDirection=list(self.view + 0.5 * self.up))
        return np.reshape(np.array(img[2], dtype=np.uint8), (H, W, 4))[..., :3]


def record(uid, meta, n, P, cam):
    scene = sg.Scene(uid, meta["up"], n, "rack")
    scene.uid = uid
    rec = Recorder(cam, scene.up)
    scene.on_step = rec
    r = scene.trial(P)
    return rec.frames, rec.phases, r


def annotate(img, name, gripper, phase, t, outcome=None, scale=1.0):
    im = Image.fromarray(img).resize((int(W * scale), int(H * scale)), Image.LANCZOS) if scale != 1 else Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f1 = ImageFont.truetype(FONT_BOLD, int(17 * scale))
    f2 = ImageFont.truetype(FONT, int(15 * scale))
    pad = int(8 * scale)
    d.text((pad, pad), f"{name}  ·  {gripper}", font=f1, fill=(20, 20, 20))
    d.text((pad, pad + int(22 * scale)), phase, font=f2, fill=(70, 70, 70))
    tt = f"{t:4.1f} s"
    d.text((im.width - pad - d.textlength(tt, font=f2), pad), tt, font=f2, fill=(110, 110, 110))
    if outcome is not None:
        f3 = ImageFont.truetype(FONT_BOLD, int(19 * scale))
        txt = outcome
        tw = d.textlength(txt, font=f3)
        x0, y1 = (im.width - tw) / 2 - pad, im.height - pad
        y0 = y1 - int(30 * scale)
        d.rounded_rectangle([x0, y0, x0 + tw + 2 * pad, y1], radius=int(6 * scale),
                            fill=GREEN if outcome == "held" else RED)
        d.text((x0 + pad, y0 + int(4 * scale)), txt, font=f3, fill=(255, 255, 255))
    return np.asarray(im)


def panel_frames(frames, phases, outcome, name, gripper, total, scale=1.0):
    """The run's frames with their labels, the last one held with the outcome until `total` frames."""
    out = [annotate(f, name, gripper, ph, (i + 1) / FPS, scale=scale) for i, (f, ph) in enumerate(zip(frames, phases))]
    end = annotate(frames[-1], name, gripper, "done", len(frames) / FPS, outcome=outcome, scale=scale)
    return out + [end] * (total - len(out))


class Writer:
    """H.264 through ffmpeg, yuv420p so QuickTime plays it."""

    def __init__(self, path, size, crf=20):
        self.proc = subprocess.Popen(
            [shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
             "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((sg.SIM/"objects.json").read_text())
    csv = pd.read_csv(sg.OUT/"sim_grasps.csv")
    csv = csv[(csv.model == "rack") & (csv["rank"] == 1)].set_index(["uid", "gripper"])
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="simvid_"))
    p.connect(p.DIRECT)
    # every run is padded to the longest possible one, so that the videos and the grid stay in step
    full = int(np.ceil((0.05 + 1.0 + 0.125 + max(1.5, (sg.TH_OPEN - sg.TH_CLOSED) / sg.CLOSE_SPEED + 0.5)
                        + 1.0 + 0.5 + 1.0 + 0.25) * FPS)) + int(END_HOLD * FPS)
    checks, smalls = [], []
    for uid, m in meta.items():
        z = np.load(sg.SIM/f"{uid}.npz", allow_pickle=True)
        poses = {n: np.asarray(z[f"{n}f|poses"][0], float) for n in (4, 2)}
        probe = sg.Scene(uid, m["up"], 4, "rack")
        probe.uid = uid
        cam = camera(probe, list(poses.values()))
        runs = {}
        for n in (4, 2):
            frames, phases, r = record(uid, m, n, poses[n], cam)
            want = csv.loc[(uid, f"{n}-finger")].outcome
            checks.append(dict(object=m["name"], gripper=f"{n}-finger", video=r["outcome"], csv=want,
                               match=r["outcome"] == want))
            runs[n] = (frames, phases, r["outcome"])
        panels = {n: panel_frames(*runs[n], m["name"], f"{n}-finger", full) for n in (4, 2)}
        sep = np.full((H, 4, 3), 60, np.uint8)
        w = Writer(OUT/f"{m['name'].replace(' ', '_')}.mp4", (2 * W + 4, H))
        for a, b in zip(panels[4], panels[2]):
            w.write(np.hstack([a, sep, b]))
        w.close()
        for n in (4, 2):                                  # small copies for the grid
            path = tmp/f"{len(smalls):02d}.mp4"
            ws = Writer(path, SMALL, crf=16)
            for f in panel_frames(*runs[n], m["name"], f"{n}-finger", full, scale=SMALL[0] / W):
                ws.write(f)
            ws.close()
            smalls.append(path)
        print(f"{m['name']:16s} 4-finger {runs[4][2]:20s} 2-finger {runs[2][2]}", flush=True)
    p.disconnect()

    cols = 4
    layout = "|".join(f"{(i % cols) * SMALL[0]}_{(i // cols) * SMALL[1]}" for i in range(len(smalls)))
    cmd = [shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error"]
    for s in smalls:
        cmd += ["-i", str(s)]
    cmd += ["-filter_complex", f"xstack=inputs={len(smalls)}:layout={layout}:fill=white", "-c:v", "libx264",
            "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p", "-r", str(FPS), "-movflags", "+faststart",
            str(OUT/"all.mp4")]
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmp)

    df = pd.DataFrame(checks)
    print("\n" + df.to_string(index=False))
    print(f"\n{df.match.sum()} of {len(df)} outcomes match sim_grasps.csv")
    for f in sorted(OUT.glob("*.mp4")):
        print(f"{f.relative_to(sg.ROOT)}  {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
