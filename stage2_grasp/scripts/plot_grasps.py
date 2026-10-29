"""Figures for evaluate_grasps.py, from its CSV and the saved poses (no re-planning).

    grasp_check.png      per method, how many of the 10 top-scored grasps the real fingers close on
                         validly, for the objects stage 1's rule sends to each gripper
    grasp_check_4f.png   the best 4-finger grasp, old square box against composed, drawn with the
                         real gripper (STLs) closed to where each finger first touches
    graspgenx_unseen_objects_4f.png / .csv
                         the 10 YCB objects of stage2_grasp_4finger.ipynb, redone with the composed
                         planner: the grasp the pipeline executes, drawn the same way

    /opt/anaconda3/envs/graspgenx/bin/python stage2_grasp/scripts/plot_grasps.py [--unseen]
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


UNSEEN = ["002_master_chef_can", "005_tomato_soup_can", "006_mustard_bottle", "011_banana", "021_bleach_cleanser",
          "025_mug", "035_power_drill", "037_scissors", "048_hammer", "055_baseball"]


def forks(pose, res):
    """A 4-finger grasp as two 2-finger forks (GraspGenX's outline, as in graspgenx_unseen_objects.png),
    closed on the object: each finger on the side where it first touches (finray.Checker), from its root
    53 mm in front of the mounting face (four_finger.BAND) to its real fingertip. Returns, per pair, the
    U (tip, root, centre, root, tip) and the contact points on it, and the stem."""
    tip = FING.vertices[np.argmax(FING.vertices[:, 2])] * 1e-3
    Pi, lo = np.linalg.inv(pose), 0.053
    world = lambda q: np.asarray(q, float) @ pose[:3, :3].T + pose[:3, 3]

    def finger(a):
        side = np.array([np.cos(np.radians(a)), np.sin(np.radians(a)), 0.0])
        T = finray.finger_pose(res.get("angles", {}).get(a, finray.THETA_OPEN), a)
        t = T[:3, :3] @ tip + T[:3, 3]                                 # its real fingertip, gripper frame
        p = res.get("points", {}).get(a)
        q = Pi[:3, :3] @ p + Pi[:3, 3] if p is not None else t
        x = side * (q @ side)                                          # how far out along its side
        return x + [0, 0, t[2]], x + [0, 0, lo], (x + [0, 0, q[2]] if p is not None else None)

    pairs = []
    for a, b in ((0, 180), (90, 270)):
        (ta, ra, ca), (tb, rb, cb) = finger(a), finger(b)
        pairs.append((world([ta, ra, [0, 0, lo], rb, tb]), world([c for c in (ca, cb) if c is not None]).reshape(-1, 3)))
    return pairs, world([[0, 0, 0], [0, 0, lo]])


def view_for(approach, tips):
    """(elev, azim) that shows both forks from the side, with their four fingers apart across the image:
    within 30 deg of square to the approach, as close to the usual 20 deg elevation as the spread allows."""
    best = None
    for el in (0, 10, 20, 30, 40):
        for az in range(0, 360, 5):
            e, z = np.radians(el), np.radians(az)
            c = np.array([np.cos(e) * np.cos(z), np.cos(e) * np.sin(z), np.sin(e)])
            if abs(c @ approach) > 0.5:
                continue
            right = np.array([-np.sin(z), np.cos(z), 0.0])
            up = np.cross(c, right)
            # the fingers run along the approach, so spread them across its direction in the image
            across = np.array([-(approach @ up), approach @ right])
            x = np.c_[tips @ right, tips @ up] @ (across / np.linalg.norm(across))
            spread = min(abs(x[i] - x[j]) for i in range(4) for j in range(i + 1, 4))
            score = spread - 5e-4 * abs(el - 20)
            if best is None or score > best[0]:
                best = (score, el, az)
    return best[1], best[2]


def unseen_4f(df):
    """The 10 YCB objects of the original 4-finger figure (stage2_grasp_4finger.ipynb), drawn the same way,
    now with the fixed planner. Shown: the grasp the pipeline executes, the best-scored one in the planner's
    top 100 that the real fingers close on validly (else its #1), as its two 2-finger pinches."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[df.method == "4f composed"].set_index("id")
    fig = plt.figure(figsize=(15, 7.2))
    rows = []
    for i, name in enumerate(UNSEEN):
        r = d.loc[f"ycb_{name}"]
        z = np.load(RAW/f"ycb_{name}.npz", allow_pickle=True)
        mesh = objects.ycb(name)
        mesh.apply_translation(-z["centre"])
        pts = np.asarray(trimesh.sample.sample_surface(mesh, 5000, seed=0)[0])
        poses, valid, conf = z["4f composed|poses"], z["4f composed|valid"], z["4f composed|conf"]
        k = int(np.argmax(valid)) if valid.any() else 0
        pairs, stem = forks(poses[k], finray.Checker(mesh, 4).check(poses[k]))

        ax = fig.add_subplot(2, 5, i + 1, projection="3d")
        ax.computed_zorder = False
        ax.scatter(*pts.T, s=2.0, c="0.55", alpha=.45, linewidths=0)
        ax.plot(*stem.T, color="0.3", lw=3.0, solid_capstyle="round")
        for (u, cpts), col in zip(pairs, ("#1f6fd0", "#e8a33d")):
            ax.plot(*u.T, color=col, lw=3.0, solid_capstyle="round", solid_joinstyle="round")
            ax.scatter(*cpts.T, s=46, c=col, edgecolors="white", linewidths=1.0, depthshade=False)
        allp = np.vstack([pts, pairs[0][0], pairs[1][0], stem])
        c, h = (allp.min(0) + allp.max(0)) / 2, (allp.max(0) - allp.min(0)).max() / 2
        for setter, cc in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), c):
            setter(cc - h, cc + h)
        ax.set_box_aspect((1, 1, 1), zoom=1.12)
        el, az = view_for(poses[k][:3, 2], np.array([u[j] for u, _ in pairs for j in (0, -1)]))
        ax.view_init(elev=el, azim=az); ax.set_proj_type("ortho"); ax.set_axis_off()
        ax.set_title(r.object, fontsize=11.5, pad=0)
        verdict = (f"valid  ·  planner's #{k + 1}  ·  score {conf[k]:.2f}" if valid.any()
                   else "no valid 4-finger grasp")
        ax.text2D(.5, .99, verdict, ha="center", va="top", fontsize=9.5, transform=ax.transAxes,
                  color="#1e7b34" if valid.any() else "#be1e1e")
        rows.append(dict(object=r.object, short_mm=r.short_mm, stage1_rule=r.rule, shown_rank=k + 1,
                         score=round(float(conf[k]), 3), valid=bool(valid.any()), top1_valid=r.top1_valid,
                         valid_top10=r.valid_top10, why_top1_fails=r.why_top1_fails))
    fig.suptitle("GraspGenX with our 4-finger fin-ray gripper, on 10 YCB objects it never saw",
                 fontsize=14, fontweight="bold", y=.99)
    fig.text(.5, .935, "Each 4-finger grasp is two 2-finger pinches at 90°, blue and orange, on one approach (grey), "
             "closed to where the fingers touch (dots).  Shown: the grasp the pipeline executes.", ha="center", fontsize=10.5, color="0.3")
    fig.subplots_adjust(top=.88, bottom=.0, left=.0, right=1, wspace=0, hspace=.12)
    fig.savefig(OUT/"graspgenx_unseen_objects_4f.png", dpi=150, facecolor="white")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/"graspgenx_unseen_objects_4f.csv", index=False)


def main():
    df = pd.read_csv(OUT/"grasp_check.csv")
    if "--unseen" in sys.argv:
        unseen_4f(df)
        print("saved", (OUT/"graspgenx_unseen_objects_4f.png").relative_to(ROOT))
        return
    summary_plot(df)
    four = df[(df.method == "4f composed") & (df.rule == "4-finger") & df.top1_valid].object.tolist()
    flat = df[(df.method == "4f composed") & (df.rule == "2-finger")].sort_values("valid_top10").object.tolist()
    collage(df, four[:5] + flat[:1])
    unseen_4f(df)
    print("saved", (OUT/"grasp_check.png").relative_to(ROOT), "and", (OUT/"grasp_check_4f.png").relative_to(ROOT))


if __name__ == "__main__":
    main()
