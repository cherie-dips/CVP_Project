import pathlib, trimesh, numpy as np, matplotlib, warnings
matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

GRIPPER = pathlib.Path(__file__).resolve().parents[2]/"data"/"gripper"
BODY  = trimesh.load(GRIPPER/"Body.stl", force="mesh")
CLAMP = trimesh.load(GRIPPER/"Correct.stl", force="mesh")
FING  = trimesh.load(GRIPPER/"Fixed_Correct_Finger.stl", force="mesh")

# measured
P_CLAMP   = np.array([40.17, 22.83, 5.2])     # pivot in clamp coords (mid-thickness)
SLOT_X    = 51.75                             # finger width-centre in clamp coords
SLOT_Y    = 4.0                               # finger root sits here, extends to -y
R_PIVOT   = 30.50                             # body: pivot radius
Z_PIVOT   = -6.19                             # body: pivot height

def finger_in_clamp():
    """Finger placed into the clamp's slot, in clamp coordinates."""
    T = np.eye(4)
    # finger x(width)->clamp x, finger z(length)->clamp -y, finger y(thick)->clamp z
    T[:3, :3] = np.array([[1, 0, 0],
                          [0, 0, -1],
                          [0, 1, 0]], float)
    T[:3, 3] = [SLOT_X, SLOT_Y, P_CLAMP[2] - 5.0]   # centre the 10 mm thickness on the plate
    return T

def clamp_to_body(sign, theta_deg, about_deg):
    """sign=-1 puts the finger radially inward of the pivot, +1 outward."""
    ey = np.array([0, 0, -1.0])                  # clamp +y -> body -z
    ez = np.array([0, sign*1.0, 0])              # clamp +z -> body tangential
    ex = np.cross(ey, ez)                        # right-handed
    R = np.column_stack([ex, ey, ez])
    T = np.eye(4); T[:3, :3] = R
    T[:3, 3] = np.array([R_PIVOT, 0, Z_PIVOT]) - R @ P_CLAMP
    # rotate about the pivot axis (body y) by theta, then around z to the arm
    piv = np.array([R_PIVOT, 0, Z_PIVOT])
    Rt = trimesh.transformations.rotation_matrix(np.radians(theta_deg), [0, 1, 0], piv)
    Rz = trimesh.transformations.rotation_matrix(np.radians(about_deg), [0, 0, 1])
    return Rz @ Rt @ T

def build(sign, theta, n_fingers=4):
    parts = [BODY.copy()]
    for k in range(n_fingers):
        M = clamp_to_body(sign, theta, 90*k)
        c = CLAMP.copy(); c.apply_transform(M); parts.append(c)
        f = FING.copy();  f.apply_transform(M @ finger_in_clamp()); parts.append(f)
    return parts

if __name__ == "__main__":
    fig = plt.figure(figsize=(14, 7))
    for col, sign in enumerate([-1, +1]):
        parts = build(sign, 0.0)
        pts = np.vstack([np.asarray(trimesh.sample.sample_surface(p, 40000)[0]) for p in parts])
        for row, (el, az, t) in enumerate([(12, -60, "angled"), (0, -90, "side (+x right)")]):
            ax = fig.add_subplot(2, 2, row*2+col+1, projection="3d")
            o = np.argsort(pts[:, 1])
            ax.scatter(*pts[o].T, s=0.7, c=pts[o][:, 2], cmap="viridis", linewidths=0)
            L = 95
            ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-60, 130)
            ax.set_box_aspect((1, 1, 1.1), zoom=1.3); ax.view_init(elev=el, azim=az)
            ax.set_title(f"fingers {'inward' if sign<0 else 'outward'} — {t}", fontsize=10)
            ax.set_axis_off()
    fig.tight_layout(); fig.savefig("/tmp/assembly.png", dpi=125, facecolor="white")
    print("rendered both mountings")
