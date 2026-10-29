"""URDFs of the fin-ray gripper for PyBullet (sim_grasps.py), 2- and 4-finger, from the STLs.

The assembly is assemble_gripper.py / finray.py's: the body, then a clamp with its finger on each
pivot. clamp_to_body(+1, theta, a) = Rz(a) . Trans(pivot) . Roty(theta) . Trans(-pivot) . T, so each
clamp is a revolute joint at Rz(a) . pivot about Rz(a) . y = (-sin a, cos a, 0), and the clamp and
finger meshes are baked into that joint's frame once (the same meshes serve every pivot). The base
link is GraspGenX's grasp frame: metres, origin at the mounting face, +z along the fingers, +x the
first closing axis. Joint angle = clamp angle: theta_open 18.27 deg (fingertips 148.1 mm apart) to
theta_closed -23.34 deg (opposing tips meet).

Collision: the fin-ray finger as its convex hull, which is its triangular envelope, so the hull's
inner face is the contact beam; body and clamp by convex decomposition (p.vhacd), since their hulls
are 6x and much larger than the parts.

Two variants per gripper:
    finray_Nf.urdf        clamp and finger one rigid link per pivot ("independent" model)
    finray_Nf_rack.urdf   the finger on a prismatic slide in its clamp, along the contact beam's
                          normal, for a spring that stands in for the fin-ray's give ("rack" model)

    /opt/anaconda3/envs/graspgenx/bin/python stage2_grasp/scripts/sim_gripper_urdf.py
        -> data/sim/gripper/finray_2f.urdf, finray_4f.urdf and their meshes
"""
import pathlib, sys

import numpy as np, pybullet as p, trimesh

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"stage2_grasp"/"scripts"))
import finray
from assemble_gripper import BODY, CLAMP, FING, P_CLAMP, R_PIVOT, Z_PIVOT, finger_in_clamp

OUT = ROOT/"data"/"sim"/"gripper"
TORQUE = 0.6                # N m per clamp: ~8 N at a contact 75 mm from the pivot
MASS = {"body": 0.30, "clamp": 0.08, "finger": 0.05}     # kg; "clamp" is clamp + finger. The finger
                                                          # is heavier than its ~6 g of TPU so that its
                                                          # spring (sim_grasps.py) integrates stably
SPRING_TRAVEL = 0.02        # m the finger can be pushed back on its spring slide (rack model)
COLOURS = {"body": "0.27 0.27 0.30 1", "clamp": "0.47 0.47 0.51 1",
           0: "0.12 0.44 0.82 1", 180: "0.12 0.44 0.82 1", 90: "0.91 0.55 0.16 1", 270: "0.91 0.55 0.16 1"}


def _trans(v):
    M = np.eye(4); M[:3, 3] = v
    return M


def link_meshes():
    """Body in the grasp frame; clamp and finger in their joint frame. Metres."""
    piv = np.array([R_PIVOT, 0, Z_PIVOT])
    ey, ez = np.array([0, 0, -1.0]), np.array([0, 1.0, 0])
    R = np.column_stack([np.cross(ey, ez), ey, ez])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = piv - R @ P_CLAMP          # clamp_to_body's constant part
    in_joint = _trans(-piv) @ T
    body, clamp, finger = BODY.copy(), CLAMP.copy(), FING.copy()
    body.apply_translation([0, 0, -finray.Z_BASE])
    clamp.apply_transform(in_joint)
    finger.apply_transform(in_joint @ finger_in_clamp())
    for m in (body, clamp, finger):
        m.apply_scale(1e-3)
    return body, clamp, finger


def _vhacd(src, dst, resolution=300000):
    if dst.exists():
        return
    p.vhacd(str(src), str(dst), str(dst.with_suffix(".log")), resolution=resolution, concavity=0.001,
            maxNumVerticesPerCH=64, minVolumePerCH=1e-5, gamma=0.0005)


def _inertial(mesh, mass, at_origin=False):
    """at_origin keeps the inertial frame on the link frame: PyBullet reports and sets a base's pose
    at its inertial frame, and the base has to sit exactly on the grasp frame."""
    hull = mesh.convex_hull
    I = hull.moment_inertia * mass / hull.volume                          # trimesh: unit density
    c = np.zeros(3) if at_origin else hull.center_mass
    return (f'<inertial><origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}"/><mass value="{mass}"/>'
            f'<inertia ixx="{I[0, 0]:.3e}" ixy="{I[0, 1]:.3e}" ixz="{I[0, 2]:.3e}" iyy="{I[1, 1]:.3e}" '
            f'iyz="{I[1, 2]:.3e}" izz="{I[2, 2]:.3e}"/></inertial>')


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    p.connect(p.DIRECT)
    body, clamp, finger = link_meshes()
    for name, m in (("body", body), ("clamp", clamp), ("finger", finger)):
        m.export(OUT/f"{name}.obj")
    finger.convex_hull.export(OUT/"finger_hull.obj")
    _vhacd(OUT/"body.obj", OUT/"body_vhacd.obj")
    _vhacd(OUT/"clamp.obj", OUT/"clamp_vhacd.obj")
    p.disconnect()
    arm = trimesh.util.concatenate([clamp, finger])
    hull = finger.convex_hull
    beam = hull.facets_normal[np.argmin(hull.facets_normal[:, 0])]        # contact beam: faces the axis (-x)
    zp = (Z_PIVOT - finray.Z_BASE) * 1e-3
    lo, hi = np.radians(finray.THETA_CLOSED), np.radians(finray.THETA_OPEN)
    vis = lambda mesh, name, colour: (f'<visual><geometry><mesh filename="{mesh}"/></geometry><material name="{name}">'
                                      f'<color rgba="{colour}"/></material></visual>')
    col = lambda mesh: f'<collision><geometry><mesh filename="{mesh}"/></geometry></collision>'
    for n in (2, 4):
        for rack in (False, True):
            xml = [f'<?xml version="1.0"?>\n<robot name="finray_{n}f{"_rack" if rack else ""}">',
                   '<link name="body">' + _inertial(body, MASS["body"], at_origin=True) +
                   vis("body.obj", "body", COLOURS["body"]) + col("body_vhacd.obj") + '</link>']
            for a in finray.abouts(n):
                c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
                clamp_link = f"clamp_{a}" if rack else f"arm_{a}"
                if rack:   # the finger on a spring slide, pushed back along the contact beam's normal
                    xml.append(f'<link name="{clamp_link}">' + _inertial(clamp, MASS["clamp"] - MASS["finger"]) +
                               vis("clamp.obj", f"clamp_{a}", COLOURS["clamp"]) + col("clamp_vhacd.obj") + '</link>')
                    xml.append(f'<link name="finger_{a}">' + _inertial(finger, MASS["finger"]) +
                               vis("finger.obj", f"finger_{a}", COLOURS[a]) + col("finger_hull.obj") + '</link>')
                    xml.append(f'<joint name="spring_{a}" type="prismatic"><parent link="{clamp_link}"/>'
                               f'<child link="finger_{a}"/><origin xyz="0 0 0" rpy="0 0 0"/>'
                               f'<axis xyz="{-beam[0]:.5f} {-beam[1]:.5f} {-beam[2]:.5f}"/>'
                               f'<limit lower="0" upper="{SPRING_TRAVEL}" effort="200" velocity="1.0"/>'
                               '<dynamics damping="0.0" friction="0.0"/></joint>')
                else:
                    xml.append(f'<link name="{clamp_link}">' + _inertial(arm, MASS["clamp"]) +
                               vis("clamp.obj", f"clamp_{a}", COLOURS["clamp"]) +
                               vis("finger.obj", f"finger_{a}", COLOURS[a]) +
                               col("clamp_vhacd.obj") + col("finger_hull.obj") + '</link>')
                xml.append(
                    f'<joint name="clamp_{a}" type="revolute"><parent link="body"/><child link="{clamp_link}"/>'
                    f'<origin xyz="{R_PIVOT * 1e-3 * c:.6f} {R_PIVOT * 1e-3 * s:.6f} {zp:.6f}" '
                    f'rpy="0 0 {np.radians(a):.6f}"/><axis xyz="0 1 0"/>'
                    f'<limit lower="{lo:.5f}" upper="{hi:.5f}" effort="{TORQUE}" velocity="2.0"/>'
                    '<dynamics damping="0.01" friction="0.0"/></joint>')
            xml.append("</robot>\n")
            (OUT/f"finray_{n}f{'_rack' if rack else ''}.urdf").write_text("\n".join(xml))
    print("wrote", *(str(f.relative_to(ROOT)) for f in sorted(OUT.glob("*.urdf"))))


def check():
    """The URDF at a few clamp angles against finray.meshes: same finger placement."""
    p.connect(p.DIRECT)
    g = p.loadURDF(str(OUT/"finray_4f.urdf"), useFixedBase=True)
    tip_local = FING.vertices[np.argmax(FING.vertices[:, 2])]
    worst = 0.0
    for th in (finray.THETA_CLOSED, 0.0, finray.THETA_OPEN):
        for j in range(p.getNumJoints(g)):
            p.resetJointState(g, j, np.radians(th))
        for j, a in enumerate(finray.abouts(4)):
            pos, orn = p.getLinkState(g, j, computeForwardKinematics=True)[4:6]
            Rl = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            _, _, finger = link_meshes()
            tip_urdf = Rl @ finger.vertices[np.argmax(FING.vertices[:, 2])] + pos
            T = finray.finger_pose(th, a)
            tip_ref = T[:3, :3] @ (tip_local * 1e-3) + T[:3, 3]
            worst = max(worst, np.linalg.norm(tip_urdf - tip_ref))
    p.disconnect()
    print(f"fingertip placement vs finray.py: max error {worst * 1e3:.4f} mm")


if __name__ == "__main__":
    build()
    check()
