"""Physics check of GraspGenX's grasps in PyBullet: 8 Objaverse objects, the 10 top-scored grasps per
gripper (sim_plan.py), 4-finger then 2-finger, with the URDFs from sim_gripper_urdf.py.

Protocol, in the style of ACRONYM / GraspGen's simulated grasp labels, with no table because the
planners model none:
    1. the object floats at its planned pose, gravity off; the gripper starts open, 10 cm back
       along its approach axis
    2. approach: the gripper moves in to the grasp pose (1 s). If that pushes the object more than
       15 mm, the grasp collides: "pushed on approach"
    3. close: the full stroke at the closing speed, then 0.5 s to settle (at least 1.5 s). No finger
       on the object at the end: "squeezed out" if one touched it on the way, else "no contact"
    4. lift: gravity on along the object's down, the gripper rises 15 cm (1 s) and holds (0.5 s)
    5. shake: +-3 cm at 2 Hz along up and sideways (1 s)
    Held = the object still touches the gripper and has drifted less than 3 cm in the gripper's
    frame since the close. Otherwise "dropped on lift" or "dropped on shake".

Two models of the drive, both rigid-body (PyBullet cannot bend a fin-ray):
    rack         closer to the real gripper. One rack turns all clamps together: the clamps follow a
                 shared angle that closes at 1 rad/s while the clamps' torques add up to less than the
                 drive's 2.4 N m and backs off above it; after the close it is held (a rack does not
                 back-drive). The fin-ray's give is a spring
                 slide under each finger, along its contact beam's normal: 1 N/mm (the RA-L 2026
                 fin-ray loads a finger this size ~10 N over ~10 mm), 20 mm travel. The 2-finger
                 gripper has the same drive, so each of its two fingers gets twice the share.
    independent  lenient reference: each clamp has its own motor with the same target and 0.6 N m,
                 so one pair can keep closing after the other has stopped, and fingers are rigid.

Physics: object mass = 400 kg/m^3 x its convex hull's volume, clamped to 0.05-0.5 kg (Objaverse
meshes are rarely watertight, so their own volume is unreliable). Friction 0.6 at finger contacts
(TPU; Bullet multiplies the two bodies' coefficients, so the object gets 1.0). Collision: the object
by convex decomposition (p.vhacd), the gripper as in sim_gripper_urdf.py.

    /opt/anaconda3/envs/graspgenx/bin/python stage2_grasp/scripts/sim_grasps.py [--only uid,uid]
        -> stage2_grasp/results/sim_grasps.csv, sim_grasps.png, sim_snapshots.png
"""
import json, pathlib, sys, time

import numpy as np, pandas as pd, pybullet as p, trimesh

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"stage2_grasp"/"scripts"))
import finray
from sim_gripper_urdf import SPRING_TRAVEL, TORQUE

SIM, OUT = ROOT/"data"/"sim", ROOT/"stage2_grasp"/"results"
DT = 1 / 480
DENSITY, MASS_RANGE = 400.0, (0.05, 0.5)
MU = 0.6
BACKOFF, LIFT, SHAKE_AMP, SHAKE_HZ = 0.10, 0.15, 0.03, 2.0
PUSH_TOL, HOLD_TOL = 0.015, 0.03
MOVE_FORCE = 2000.0
CLOSE_SPEED = 1.0                       # rad/s at the clamp: the full stroke in ~0.7 s
DRIVE = 4 * TORQUE                      # N m the rack delivers, shared by the clamps on it
CLAMP_HOLD = 20.0                       # N m: a clamp holds its rack angle (rigid rack)
K_SPRING, C_SPRING = 1000.0, 3.0        # N/m, N s/m: the fin-ray's give at the contact
TH_OPEN, TH_CLOSED = np.radians(finray.THETA_OPEN), np.radians(finray.THETA_CLOSED)
UP = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
MODELS = ("rack", "independent")


def quat(R):
    """Rotation matrix -> PyBullet quaternion (x, y, z, w)."""
    q = trimesh.transformations.quaternion_from_matrix(np.r_[np.c_[R, np.zeros(3)], [[0, 0, 0, 1]]])
    return [q[1], q[2], q[3], q[0]]


def pose(body):
    pos, orn = p.getBasePositionAndOrientation(body)
    return np.array(pos), np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)


class Scene:
    """One object and one gripper in a fresh PyBullet world; trial() runs one grasp."""

    def __init__(self, uid, up, n, model):
        p.resetSimulation()
        p.setTimeStep(DT)
        p.setPhysicsEngineParameter(numSolverIterations=150, enableConeFriction=1,
                                    deterministicOverlappingPairs=1)
        self.n, self.model, self.up = n, model, np.array(UP[up], float)
        obj = SIM/"objects"/f"{uid}.obj"
        col = SIM/"objects"/f"{uid}_vhacd.obj"
        if not col.exists():
            p.vhacd(str(obj), str(col), str(col.with_suffix(".log")), resolution=300000,
                    concavity=0.0025, maxNumVerticesPerCH=64, minVolumePerCH=1e-5)
        hull = trimesh.load(obj, force="mesh").convex_hull
        self.mass = float(np.clip(DENSITY * hull.volume, *MASS_RANGE))
        self.com = hull.center_mass                       # the base frame sits on the centre of mass
        cs = p.createCollisionShape(p.GEOM_MESH, fileName=str(col), collisionFramePosition=-self.com)
        vs = p.createVisualShape(p.GEOM_MESH, fileName=str(obj), visualFramePosition=-self.com,
                                 rgbaColor=[0.78, 0.78, 0.78, 1])
        self.obj = p.createMultiBody(self.mass, cs, vs, basePosition=self.com)
        p.changeDynamics(self.obj, -1, lateralFriction=1.0, spinningFriction=0.002, rollingFriction=0.0,
                         restitution=0.0)
        urdf = SIM/"gripper"/f"finray_{n}f{'_rack' if model == 'rack' else ''}.urdf"
        self.grip = p.loadURDF(str(urdf), [0, 0, 1], useFixedBase=False, flags=p.URDF_USE_INERTIA_FROM_FILE)
        info = [p.getJointInfo(self.grip, j) for j in range(p.getNumJoints(self.grip))]
        self.clamps = [i[0] for i in info if i[1].decode().startswith("clamp_")]
        self.springs = [i[0] for i in info if i[1].decode().startswith("spring_")]
        self.fingers = self.springs or self.clamps           # the link that carries each finger
        for link in [-1] + [i[0] for i in info]:
            p.changeDynamics(self.grip, link, lateralFriction=MU, spinningFriction=0.002, restitution=0.0)
        for j in self.springs:                                # free slides: the spring is applied by hand
            p.setJointMotorControl2(self.grip, j, p.VELOCITY_CONTROL, force=0)
        self.cid = p.createConstraint(self.grip, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1])
        self.rack, self.closing = TH_OPEN, False
        self.phase, self.goal, self.on_step = "", None, None    # for sim_videos.py; no effect on physics

    def _move(self, pos, orn):
        p.changeConstraint(self.cid, list(pos), orn, maxForce=MOVE_FORCE)

    def _hold_clamps(self):
        for j in self.clamps:
            if self.model == "rack":
                p.setJointMotorControl2(self.grip, j, p.POSITION_CONTROL, targetPosition=self.rack, force=CLAMP_HOLD)
            elif self.closing:
                p.setJointMotorControl2(self.grip, j, p.POSITION_CONTROL, targetPosition=TH_CLOSED, force=TORQUE,
                                        maxVelocity=CLOSE_SPEED)
            else:
                p.setJointMotorControl2(self.grip, j, p.POSITION_CONTROL, targetPosition=TH_OPEN, force=TORQUE)

    def _tick(self):
        if self.model != "rack":
            return
        if self.closing:                  # a torque-limited drive: advance below its torque, yield above
            load = sum(abs(p.getJointState(self.grip, j)[3]) for j in self.clamps)
            if load < DRIVE and self.rack > TH_CLOSED:
                self.rack = max(TH_CLOSED, self.rack - CLOSE_SPEED * DT)
                self._hold_clamps()
            elif load > 1.1 * DRIVE:
                self.rack = min(TH_OPEN, self.rack + 0.25 * CLOSE_SPEED * DT)
                self._hold_clamps()
        s = p.getJointStates(self.grip, self.springs)
        p.setJointMotorControlArray(self.grip, self.springs, p.TORQUE_CONTROL,
                                    forces=[-K_SPRING * q - C_SPRING * v for q, v, _, _ in s])

    def _run(self, seconds, path=None):
        """Step for `seconds`; path(f) gives the gripper pose at fraction f of the way."""
        steps = int(round(seconds / DT))
        for i in range(steps):
            if path is not None:
                self._move(*path((i + 1) / steps))
            self._tick()
            p.stepSimulation()
            if self.on_step is not None:
                self.on_step(self)

    def _rel(self):
        """The object's centre in the gripper's frame."""
        po, _ = pose(self.obj)
        pg, Rg = pose(self.grip)
        return Rg.T @ (po - pg)

    def _touching(self):
        return [len(p.getContactPoints(self.grip, self.obj, linkIndexA=j)) > 0 for j in self.fingers]

    def _held(self):
        return len(p.getContactPoints(self.grip, self.obj)) > 0

    def trial(self, P, snapshot=None):
        p.setGravity(0, 0, 0)
        p.resetBasePositionAndOrientation(self.obj, self.com, [0, 0, 0, 1])
        p.resetBaseVelocity(self.obj, [0, 0, 0], [0, 0, 0])
        R, goal = P[:3, :3], P[:3, 3]
        orn = quat(R)
        start = goal - BACKOFF * R[:, 2]
        self.goal, self.phase = goal, "approach"
        p.resetBasePositionAndOrientation(self.grip, start, orn)
        p.resetBaseVelocity(self.grip, [0, 0, 0], [0, 0, 0])
        for j in self.clamps:
            p.resetJointState(self.grip, j, TH_OPEN)
        for j in self.springs:
            p.resetJointState(self.grip, j, 0.0)
        self.rack, self.closing = TH_OPEN, False
        self._hold_clamps()
        self._move(start, orn)
        self._run(0.05)
        out = dict(push_mm=np.nan, fingers_touching=0, drift_mm=np.nan, pair_gap_deg=np.nan, spring_mm=np.nan)

        self._run(1.0, lambda f: (start + (goal - start) * f, orn))
        self._run(0.125)
        out["push_mm"] = 1e3 * np.linalg.norm(pose(self.obj)[0] - self.com)
        if out["push_mm"] > PUSH_TOL * 1e3:
            return dict(out, outcome="pushed on approach", success=False)

        self.closing = True
        self._hold_clamps()
        touched, stroke = False, (TH_OPEN - TH_CLOSED) / CLOSE_SPEED
        for k in range(int(round(max(1.5, stroke + 0.5) / 0.05))):
            self.phase = "close" if k * 0.05 < stroke else "settle"
            self._run(0.05)                                   # the full stroke, then settle
            touched = touched or any(self._touching())
        self.closing = False
        touching = self._touching()
        out["fingers_touching"] = int(sum(touching))
        angles = [p.getJointState(self.grip, j)[0] for j in self.clamps]
        if self.n == 4:
            out["pair_gap_deg"] = np.degrees(abs((angles[0] + angles[2]) / 2 - (angles[1] + angles[3]) / 2))
        if self.springs:
            out["spring_mm"] = 1e3 * max(p.getJointState(self.grip, j)[0] for j in self.springs)
        if not any(touching):
            return dict(out, outcome="squeezed out" if touched else "no contact", success=False)
        rel0 = self._rel()

        p.setGravity(*(-9.81 * self.up))
        self.phase = "lift"
        self._run(1.0, lambda f: (goal + LIFT * self.up * f, orn))
        self.phase = "hold"
        self._run(0.5)
        top = goal + LIFT * self.up
        if snapshot is not None:
            snapshot(self)
        out["drift_mm"] = 1e3 * np.linalg.norm(self._rel() - rel0)
        if not self._held() or out["drift_mm"] > HOLD_TOL * 1e3:
            return dict(out, outcome="dropped on lift", success=False)

        side = np.cross(self.up, R[:, 2])
        side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-6 else R[:, 0]
        wobble = (self.up + side) / np.sqrt(2)
        self.phase = "shake"
        self._run(1.0, lambda f: (top + SHAKE_AMP * np.sin(2 * np.pi * SHAKE_HZ * f) * wobble, orn))
        self._run(0.25, lambda f: (top, orn))
        out["drift_mm"] = 1e3 * np.linalg.norm(self._rel() - rel0)
        if not self._held() or out["drift_mm"] > HOLD_TOL * 1e3:
            return dict(out, outcome="dropped on shake", success=False)
        return dict(out, outcome="held", success=True)


def render(scene, size=360):
    """TinyRenderer view of the gripper's fingers from its side, the object's up kept up."""
    pg, Rg = pose(scene.grip)
    d = Rg[:, 0] + 0.6 * Rg[:, 1]
    d = d - (d @ scene.up) * scene.up
    d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-6 else np.cross(scene.up, [1, 0, 0])
    target = pg + 0.11 * Rg[:, 2]
    eye = target + 0.42 * (np.cos(np.radians(20)) * d + np.sin(np.radians(20)) * scene.up)
    view = p.computeViewMatrix(list(eye), list(target), list(scene.up))
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.02, farVal=2.0)
    img = p.getCameraImage(size, size, view, proj, renderer=p.ER_TINY_RENDERER, shadow=0,
                           lightDirection=list(eye - target + 0.3 * scene.up))
    return np.reshape(np.array(img[2], dtype=np.uint8), (size, size, 4))[..., :3]


def run(only=None):
    meta = json.loads((SIM/"objects.json").read_text())
    rows, shots = [], {}
    p.connect(p.DIRECT)
    for uid, m in meta.items():
        if only and uid not in only:
            continue
        z = np.load(SIM/f"{uid}.npz", allow_pickle=True)
        for model in MODELS:
            for g, n in (("4f", 4), ("2f", 2)):
                scene = Scene(uid, m["up"], n, model)
                t = time.time()
                for rank, P in enumerate(z[f"{g}|poses"]):
                    snap = None
                    if rank == 0 and model == "rack":
                        snap = lambda s, key=(m["name"], n): shots.__setitem__(key, render(s))
                    r = scene.trial(np.asarray(P, float), snapshot=snap)
                    rows.append(dict(object=m["name"], uid=uid, label=m["label"], model=model, gripper=f"{n}-finger",
                                     rank=rank + 1, planner_score=round(float(z[f"{g}|conf"][rank]), 3),
                                     checker_valid=bool(z[f"{g}|valid"][rank]),
                                     checker_reason=str(z[f"{g}|reason"][rank]), sim_success=r["success"],
                                     outcome=r["outcome"], fingers_touching=r["fingers_touching"],
                                     push_mm=round(r["push_mm"], 1), drift_mm=round(r["drift_mm"], 1),
                                     pair_gap_deg=round(r["pair_gap_deg"], 1), spring_mm=round(r["spring_mm"], 1),
                                     mass_kg=round(scene.mass, 3)))
                done = [r for r in rows if r["uid"] == uid and r["gripper"] == f"{n}-finger" and r["model"] == model]
                print(f"{m['name']:16s} {model:11s} {n}-finger  held {sum(r['sim_success'] for r in done)}/{len(done)}  "
                      f"({', '.join(sorted({r['outcome'] for r in done}))})  mass {scene.mass:.2f} kg  "
                      f"{time.time() - t:.0f}s", flush=True)
    p.disconnect()
    return pd.DataFrame(rows), shots


def figures(df, shots):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(df.object))
    lab = df.drop_duplicates("object").set_index("object").label.reindex(names)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True)
    x = np.arange(len(names))
    titles = {"rack": "One rack drives all clamps, fingers on 1 N/mm springs (closer to the real gripper)",
              "independent": "Each clamp on its own motor, rigid fingers (lenient reference)"}
    for ax, model in zip(axes, MODELS):
        d0 = df[df.model == model]
        for j, (gr, col) in enumerate((("4-finger", "#1f6fd0"), ("2-finger", "#e8a33d"))):
            d = d0[d0.gripper == gr].groupby("object")
            held = d.sim_success.mean().reindex(names)
            valid = d.checker_valid.mean().reindex(names)
            xs = x + (j - 0.5) * 0.38
            ax.bar(xs, held, width=0.36, color=col,
                   label=f"{gr}: held in simulation ({d0[d0.gripper == gr].sim_success.mean():.0%} overall)")
            ax.plot(xs, valid, "_", color="k", ms=15, mew=1.8)
            for xi, v in zip(xs, held):
                ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)
        ax.plot([], [], "_", color="k", ms=15, mew=1.8, label="share valid by the geometric check (finray.py)")
        ax.axvline(3.5, color=".8", lw=.8)
        ax.set_ylim(0, 1.3); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_ylabel("top-10 GraspGenX grasps")
        ax.set_title(titles[model], loc="left", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper center", ncol=3, frameon=False)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"{nm}\n(rule: {l})" for nm, l in zip(names, lab)], fontsize=8)
    fig.suptitle("PyBullet: GraspGenX's top 10 grasps per object, closed, lifted 15 cm and shaken",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT/"sim_grasps.png", dpi=150, facecolor="white")
    plt.close(fig)

    if shots:
        fig, axes = plt.subplots(2, len(names), figsize=(2.2 * len(names), 4.9))
        for c, nm in enumerate(names):
            for r, n in enumerate((4, 2)):
                a = axes[r, c]
                if (nm, n) in shots:
                    a.imshow(shots[(nm, n)])
                row = df[(df.object == nm) & (df.model == "rack") & (df.gripper == f"{n}-finger") & (df["rank"] == 1)]
                row = row.iloc[0]
                a.set_title(f"{nm}\n{n}-finger #1: {row.outcome}", fontsize=7.5,
                            color="#1e7b34" if row.sim_success else "#be1e1e")
                a.axis("off")
        fig.suptitle("Each planner's top grasp after the lift, rack model (before the shake)", fontsize=10,
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT/"sim_snapshots.png", dpi=130, facecolor="white")
        plt.close(fig)


def summary(df):
    for model in MODELS:
        print(f"\n== {model}")
        for gr in ("4-finger", "2-finger"):
            d = df[(df.model == model) & (df.gripper == gr)]
            if not len(d):
                continue
            print(f"{gr}: held {d.sim_success.mean():.0%} of {len(d)}  |  rule says 4-finger "
                  f"{d[d.label == '4-finger'].sim_success.mean():.0%}, rule says 2-finger "
                  f"{d[d.label == '2-finger'].sim_success.mean():.0%}  |  top-1 held "
                  f"{d[d['rank'] == 1].sim_success.mean():.0%}")
            ok = d.checker_valid
            print(f"   checker valid {ok.mean():.0%}: held {d[ok].sim_success.mean():.0%} of valid, "
                  f"{d[~ok].sim_success.mean():.0%} of invalid; agreement {(ok == d.sim_success).mean():.0%}")
            print("   outcomes:", d.outcome.value_counts().to_dict())


if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    df, shots = run(only)
    if not only:
        df.to_csv(OUT/"sim_grasps.csv", index=False)
        figures(df, shots)
    summary(df)
