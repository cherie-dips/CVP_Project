"""Our fin-ray gripper as geometry, from the STLs, and a kinematic check of a grasp against an object.

The gripper is assembled exactly as in assemble_gripper.py (body, then a clamp and a finger on each
of 2 or 4 pivots) and put in GraspGenX's grasp frame: metres, origin at the body's mounting face,
+z along the fingers, +x the first closing axis.

    check(pose)  closes the real fingers on the object, one clamp-angle step at a time:

        collides_open   something (body, clamp, finger) already intersects the object fully open
        contact angle   per finger, the clamp angle at which it first touches the object
        reaches         nothing collides open, every finger touches the object with its finger
                        (not its clamp) before the fingers meet, and -- with four fingers --
                        the two pairs grip within what the fin-ray can absorb: all four clamps
                        turn together off one rack, so the pair that grips first keeps being
                        driven until the second pair grips, and its fingers must take that
                        travel by bending. Allowed: 10 mm at the contact (the fin-ray force
                        sensing paper, RA-L 2026, loads a finger of this size to 10 mm).
        valid           reaches, and every opposing pair is antipodal: the line between its two
                        contacts lies inside both friction cones (Nguyen 1988), mu = 0.6 for TPU,
                        with surface normals averaged over 5 mm around each contact because a
                        fin-ray presses a patch, not a point. Rules out fingertips closing on an
                        edge or a corner.

This is geometry, not physics: no object weight, no table, no deformation beyond the bending
allowance. It rules grasps out; it does not prove the rest hold.
"""
import numpy as np, trimesh, fcl
from scipy.spatial import cKDTree

from assemble_gripper import BODY, CLAMP, FING, R_PIVOT, Z_PIVOT, clamp_to_body, finger_in_clamp

Z_BASE = -49.53                 # mm, the body's mounting face in assembly coordinates = grasp origin
THETA_CLOSED = -23.34           # deg, opposing fingertips touch
TIP_GAP_OPEN = 148.1            # mm between fingertips fully open: the aperture GraspGenX is given
DEFLECT_MM = 10.0
MU = 0.6                        # TPU on plastic/cardboard
PATCH = 5e-3                    # m, radius over which contact normals are averaged
SIGN = +1                       # fingers outboard of the pivot (assemble_gripper.py)


def _bvh(mesh, scale=1e-3):
    m = fcl.BVHModel()
    m.beginModel(len(mesh.vertices), len(mesh.faces))
    m.addSubModel(np.asarray(mesh.vertices, np.float64) * scale, np.asarray(mesh.faces, np.int32))
    m.endModel()
    return m


def _grasp_frame(M):
    """An assembly transform (mm) as a rigid transform in the grasp frame, for parts scaled to metres."""
    T = np.eye(4)
    T[:3, :3] = M[:3, :3]
    T[:3, 3] = (M[:3, 3] - [0, 0, Z_BASE]) * 1e-3
    return T


def finger_pose(theta, about):
    return _grasp_frame(clamp_to_body(SIGN, theta, about) @ finger_in_clamp())


def clamp_pose(theta, about):
    return _grasp_frame(clamp_to_body(SIGN, theta, about))


def tip_gap(theta):
    """Distance between opposing fingertips (mm), the fingertip being the finger's highest point
    when closed."""
    tip_local = FING.vertices[np.argmax(FING.vertices[:, 2])]
    T = clamp_to_body(SIGN, theta, 0) @ finger_in_clamp()
    return 2 * (T[:3, :3] @ tip_local + T[:3, 3])[0]


def theta_for_gap(gap_mm):
    lo, hi = THETA_CLOSED, 89.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tip_gap(mid) < gap_mm else (lo, mid)
    return (lo + hi) / 2


THETA_OPEN = theta_for_gap(TIP_GAP_OPEN)


def meshes(n_fingers, theta=None):
    """The assembled gripper as trimesh parts in the grasp frame (metres), for drawing."""
    theta = THETA_OPEN if theta is None else theta
    body = BODY.copy(); body.apply_scale(1e-3); body.apply_translation([0, 0, -Z_BASE * 1e-3])
    parts = [body]
    for about in abouts(n_fingers):
        for m, T in ((CLAMP, clamp_pose(theta, about)), (FING, finger_pose(theta, about))):
            p = m.copy(); p.apply_scale(1e-3); p.apply_transform(T); parts.append(p)
    return parts


def abouts(n_fingers):
    return (0, 180) if n_fingers == 2 else (0, 90, 180, 270)


class Checker:
    """Kinematic grasp check for one object mesh (in metres, any frame the poses are given in)."""

    _models = None

    def __init__(self, mesh, n_fingers, theta_open=None, coarse=2.0, fine=0.25):
        if Checker._models is None:
            Checker._models = {k: _bvh(m) for k, m in (("body", BODY), ("clamp", CLAMP), ("finger", FING))}
        self.n = n_fingers
        self.theta_open = THETA_OPEN if theta_open is None else theta_open
        self.coarse, self.fine = coarse, fine
        self.obj = fcl.CollisionObject(_bvh(mesh, 1.0), fcl.Transform())
        self.mesh = mesh
        self.face_tree = cKDTree(mesh.triangles_center)
        self.parts = {k: fcl.CollisionObject(m, fcl.Transform()) for k, m in Checker._models.items()}
        self.body_T = _grasp_frame(np.eye(4))
        self._cache = {}

    def _hit(self, part, T, contact=False):
        o = self.parts[part]
        o.setTransform(fcl.Transform(T[:3, :3], T[:3, 3]))
        req = fcl.CollisionRequest(num_max_contacts=8 if contact else 1, enable_contact=contact)
        res = fcl.CollisionResult()
        n = fcl.collide(o, self.obj, req, res)
        if not contact:
            return n > 0
        return n > 0, (np.mean([c.pos for c in res.contacts], 0) if n else None)

    def _poses(self, theta, about):
        key = (round(theta, 4), about)
        if key not in self._cache:
            self._cache[key] = (clamp_pose(theta, about), finger_pose(theta, about))
        return self._cache[key]

    def _first_contact(self, G, about):
        """Close one finger from open to fully closed: (angle, 'finger'|'clamp', point) at first touch."""
        def touching(theta, contact=False):
            C, F = self._poses(theta, about)
            if self._hit("clamp", G @ C):
                return "clamp", None
            hit, p = self._hit("finger", G @ F, contact=True)
            return ("finger", p) if hit else (None, None)

        prev = self.theta_open
        for theta in np.arange(self.theta_open - self.coarse, THETA_CLOSED - 1e-9, -self.coarse):
            part, _ = touching(theta)
            if part:
                for t in np.arange(prev - self.fine, theta - 1e-9, -self.fine):    # refine in the last step
                    part, p = touching(t, contact=True)
                    if part:
                        return t, part, p
                return theta, part, None
            prev = theta
        return None, None, None

    def _normal(self, p, d):
        """Inward surface normal at contact p for a finger pushing along d (object frame): the
        area-weighted mean of the face normals within PATCH, each turned to face the finger."""
        idx = self.face_tree.query_ball_point(p, PATCH)
        if not idx:
            idx = [self.face_tree.query(p)[1]]
        n = self.mesh.face_normals[idx]
        n = np.where((n @ d)[:, None] > 0, -n, n)                  # outward normals face the finger
        m = (n * self.mesh.area_faces[idx, None]).sum(0)
        return -m / (np.linalg.norm(m) + 1e-12)                       # inward: the way the finger pushes

    def _antipodal(self, p1, n1, p2, n2):
        l = p2 - p1
        if np.linalg.norm(l) < 1e-6:
            return False
        l /= np.linalg.norm(l)
        c = np.cos(np.arctan(MU))
        return bool(n1 @ l >= c and n2 @ -l >= c)

    def check(self, pose):
        """pose: 4x4 grasp pose (grasp frame -> object frame)."""
        G = np.asarray(pose, float)
        out = dict(valid=False, reaches=False, collides_open=False, contacts=0, deflection_mm=np.nan,
                   antipodal=False, reason="")
        if self._hit("body", G @ self.body_T):
            out.update(collides_open=True, reason="body hits the object")
            return out
        for about in abouts(self.n):
            C, F = self._poses(self.theta_open, about)
            if self._hit("clamp", G @ C) or self._hit("finger", G @ F):
                out.update(collides_open=True, reason="a finger hits the object fully open")
                return out

        angles, levers, points = {}, {}, {}
        for about in abouts(self.n):
            theta, part, p = self._first_contact(G, about)
            if part == "finger":
                angles[about] = theta
                points[about] = p
                if p is not None:                                       # contact point -> pivot distance
                    q = np.linalg.inv(G) @ np.r_[p, 1.0]
                    c, s = np.cos(np.radians(about)), np.sin(np.radians(about))
                    pivot = np.array([R_PIVOT * c, R_PIVOT * s, Z_PIVOT - Z_BASE]) * 1e-3
                    axis = np.array([-s, c, 0.0])                       # the pivot axis (body y, turned)
                    d = q[:3] - pivot
                    levers[about] = np.linalg.norm(d - (d @ axis) * axis) * 1e3
            elif part == "clamp":
                out["reason"] = "a clamp touches the object before its finger does"
        out["contacts"] = len(angles)
        out["angles"] = angles
        out["points"] = points                                          # first-contact point per finger
        if len(angles) < self.n:
            out["reason"] = out["reason"] or f"only {len(angles)} of {self.n} fingers reach the object"
            return out
        if self.n == 4:
            # each pair centres the object along its own axis and grips at about its mean contact angle
            pa, pb = (angles[0] + angles[180]) / 2, (angles[90] + angles[270]) / 2
            first = (0, 180) if pa > pb else (90, 270)
            lever = np.mean([levers.get(a, 60.0) for a in first])
            out["deflection_mm"] = lever * np.radians(abs(pa - pb))
            if out["deflection_mm"] > DEFLECT_MM:
                out["reason"] = f"pairs grip {abs(pa - pb):.0f} deg apart ({out['deflection_mm']:.0f} mm of bending)"
                return out
        out["reaches"] = True

        # every opposing pair must squeeze along its own contact line
        normals = {}
        for about, p in points.items():
            if p is None:
                out["reason"] = "no contact point"
                return out
            c, s = np.cos(np.radians(about)), np.sin(np.radians(about))
            d = G[:3, :3] @ np.array([-c, -s, 0.0])                      # this finger closes toward the axis
            normals[about] = self._normal(p, d)
        pairs = [(0, 180)] if self.n == 2 else [(0, 180), (90, 270)]
        ok = [self._antipodal(points[a], normals[a], points[b], normals[b]) for a, b in pairs]
        out["antipodal"] = all(ok)
        if not all(ok):
            out["reason"] = "a pair closes on surfaces that are not opposite (outside the friction cone)"
            return out
        out["valid"] = True
        return out
