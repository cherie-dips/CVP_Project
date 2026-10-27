"""Figures for evaluate_grasps.py, from its CSV and the saved poses (no re-planning).

    grasp_check.png      per method, how many of the 10 top-scored grasps the real fingers close on
                         validly, for the objects stage 1's rule sends to each gripper
    grasp_check_4f.png   the best 4-finger grasp, old square box against composed, drawn with the
                         real gripper (STLs) closed to where each finger first touches

    /opt/anaconda3/envs/graspgenx/bin/python stage2_grasp/scripts/plot_grasps.py
"""
import pathlib, sys, warnings

import numpy as np, pandas as pd, trimesh
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"stage2_grasp"/"scripts"))
import finray, objects
from assemble_gripper import BODY, CLAMP, FING

OUT = ROOT/"stage2_grasp"/"results"
RAW = ROOT/"data"/"grasp_eval"
warnings.filterwarnings("ignore")
ORDER = ["2f legacy", "2f stl", "2f tuned", "2f tuned moe", "4f legacy", "4f stl box",
         "4f composed, no rack check", "4f composed"]
LABELS = {"2f legacy": "2-finger, old box", "2f stl": "2-finger, wizard box from STLs",
          "2f tuned": "2-finger, tuned box", "2f tuned moe": "2-finger, old box + GraspMoE",
          "4f legacy": "4-finger, old square box", "4f stl box": "4-finger, square box from STLs",
          "4f composed, no rack check": "4-finger, two pinches, discriminator only",
          "4f composed": "4-finger, two pinches + one-rack check"}


def summary_plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = [m for m in ORDER if m in set(df.method)]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for j, (rule, col) in enumerate((("4-finger", "#1f6fd0"), ("2-finger", "#e8a33d"))):
        d = df[df.rule == rule].groupby("method")
        v, anyv = d.valid_top10.mean(), d.any_valid.mean()
        n = df[df.rule == rule].object.nunique()
        x = np.arange(len(methods)) + (j - 0.5) * 0.38
        ax.bar(x, [v[m] for m in methods], width=0.36, color=col,
               label=f"objects the stage 1 rule sends to 4 fingers ({n})" if rule == "4-finger"
               else f"objects it sends to 2 fingers ({n})")
        ax.plot(x, [anyv[m] for m in methods], "_", color="k", ms=14, mew=1.6)
        for xi, m in zip(x, methods):
            ax.text(xi, v[m] + 0.02, f"{v[m]:.0%}", ha="center", fontsize=7.5)
    ax.plot([], [], "_", color="k", ms=14, mew=1.6, label="share of objects with any valid grasp in the top 100")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([LABELS[m].replace(", ", ",\n") for m in methods], fontsize=8)
    ax.axvline(sum(m.startswith("2f") for m in methods) - 0.5, color=".8", lw=.8)
    ax.set_ylim(0, 1.32); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_ylabel("top-10 grasps that are valid")
    ax.set_title("Grasps closed with the real gripper (STLs): every finger touches, pairs squeeze opposite surfaces",
                 loc="left", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper center", frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT/"grasp_check.png", dpi=150, facecolor="white")
    plt.close(fig)


# ── a small painter's renderer with colours (as stage1_gripper/scripts/views.py, plus colour) ──
def render(parts, view, up, size=360, ss=2, margin=0.06):
    V, F, C, n = [], [], [], 0
    for m, rgb in parts:
        V.append(m.vertices); F.append(m.faces + n); C.append(np.tile(rgb, (len(m.faces), 1))); n += len(m.vertices)
    V, F, C = np.vstack(V), np.vstack(F), np.vstack(C).astype(float)
    mesh = trimesh.Trimesh(V, F, process=False)
    view = view / np.linalg.norm(view)
    right = np.cross(up, view); right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(view, right), view])
    v = (V - V.mean(0)) @ R.T
    lo, hi = v[:, :2].min(0), v[:, :2].max(0)
    s = (1 - 2 * margin) * size * ss / float((hi - lo).max())
    xy = (v[:, :2] - (lo + hi) / 2) * s + size * ss / 2
    xy[:, 1] = size * ss - xy[:, 1]
    light = np.array([-0.35, 0.55, 0.76]); light /= np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.abs((mesh.face_normals @ R.T) @ light)
    col = np.clip(C * shade[:, None], 0, 255).astype(np.uint8)
    order = np.argsort(v[F, 2].mean(1))
    img = Image.new("RGB", (size * ss, size * ss), "white")
    d = ImageDraw.Draw(img)
    for poly, c in zip(xy[F[order]].reshape(len(order), 6).tolist(), col[order].tolist()):
        d.polygon(poly, fill=tuple(c))
    return img.resize((size, size), Image.LANCZOS)


def gripper_at(pose, n_fingers, angles):
    """Body, clamps and fingers in the object frame, each finger at its own clamp angle."""
    def placed(mesh, T):
        m = mesh.copy(); m.apply_scale(1e-3); m.apply_transform(pose @ T); return m
    base = np.eye(4); base[2, 3] = -finray.Z_BASE * 1e-3
    parts = [(placed(BODY, base), (70, 70, 78))]
    for a in finray.abouts(n_fingers):
        th = angles.get(a, finray.THETA_OPEN)
        parts.append((placed(CLAMP, finray.clamp_pose(th, a)), (120, 120, 130)))
        parts.append((placed(FING, finray.finger_pose(th, a)), (31, 111, 208) if a in (0, 180) else (232, 140, 40)))
    return parts


def collage(df, names):
    rows = []
    for name in names:
        r = df[df.object == name].iloc[0]
        z = np.load(RAW/f"{r.id}.npz", allow_pickle=True)
        mesh = (objects.ycb(r.id[4:]) if r.source == "ycb"
                else objects.usd(ROOT/"selected_objects"/r.id[4:]/f"{r.id[4:]}.usd"))
        mesh.apply_translation(-z["centre"])
        tiles = []
        for method in ("4f legacy", "4f composed"):
            P = z[f"{method}|poses"][0]
            res = finray.Checker(mesh, 4).check(P)
            img = render([(mesh, (185, 185, 185))] + gripper_at(P, 4, res.get("angles", {})),
                         view=P[:3, 0] + 0.6 * P[:3, 1] - 0.35 * P[:3, 2], up=-P[:3, 2], size=320)
            tile = Image.new("RGB", (img.width, img.height + 34), "white")
            tile.paste(img, (0, 34))
            d = ImageDraw.Draw(tile)
            verdict = "valid" if res["valid"] else res["reason"].split(" (")[0]
            d.text((6, 3), f"{name[:28]}: {LABELS[method][10:]}", fill=(0, 0, 0))
            d.text((6, 18), verdict[:52], fill=(0, 120, 0) if res["valid"] else (190, 30, 30))
            tiles.append(tile)
        rows.append(tiles)
    W, H = rows[0][0].size
    sheet = Image.new("RGB", (2 * W + 6, len(rows) * (H + 6)), "white")
    for i, tiles in enumerate(rows):
        for j, t in enumerate(tiles):
            sheet.paste(t, (j * (W + 6), i * (H + 6)))
    sheet.save(OUT/"grasp_check_4f.png")


def main():
    df = pd.read_csv(OUT/"grasp_check.csv")
    summary_plot(df)
    four = df[(df.method == "4f composed") & (df.rule == "4-finger") & df.top1_valid].object.tolist()
    flat = df[(df.method == "4f composed") & (df.rule == "2-finger")].sort_values("valid_top10").object.tolist()
    collage(df, four[:5] + flat[:1])
    print("saved", (OUT/"grasp_check.png").relative_to(ROOT), "and", (OUT/"grasp_check_4f.png").relative_to(ROOT))


if __name__ == "__main__":
    main()
