"""Geometric features used to label Objaverse objects 2-finger vs 4-finger.

The Objaverse assets are not watertight, so face normals cannot be trusted.
Everything here is computed from surface point positions only.
"""
import numpy as np, trimesh


def features(mesh):
    pts = np.asarray(trimesh.sample.sample_surface(mesh, 20000)[0]) * 1000   # mm
    pts = pts - pts.mean(0)

    # principal axes: a0 is the longest, a2 the shortest
    _, _, V = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
    p = pts @ V.T
    ext = p.max(0) - p.min(0)
    order = np.argsort(-ext)
    p, ext = p[:, order], ext[order]

    # cross-section perpendicular to the long axis: does it roll, or has it flats?
    cs = p[:, 1:]
    r = np.linalg.norm(cs, axis=1)
    ang = np.arctan2(cs[:, 1], cs[:, 0])
    bins = np.linspace(-np.pi, np.pi, 25)
    idx = np.digitize(ang, bins) - 1
    radii = np.array([r[idx == b].max() if (idx == b).any() else np.nan
                      for b in range(24)])
    radii = radii[~np.isnan(radii)]

    return dict(
        long_mm  = round(float(ext[0]), 1),
        mid_mm   = round(float(ext[1]), 1),
        short_mm = round(float(ext[2]), 1),
        elongation  = round(float(ext[0]/ext[2]), 2),
        # 1.0 = perfectly circular cross-section (rolls); lower = has flats to pinch
        circularity = round(float(radii.min()/radii.max()), 3),
        # how square the cross-section is; 1.0 means no narrow direction across it
        cross_ratio = round(float(ext[1]/ext[2]), 2),
    )


APERTURE_MM = 148.1

def label(f):
    """Documented rule. Returns (gripper, reason).

    One criterion, measured from the surface: how circular is the cross-section
    across the object's long axis? A 2-finger fin-ray grip needs two opposing
    flats to pinch against. Round cross-sections have none -- the object rolls
    out of a two-point pinch -- and that is what the extra pair of fingers is
    for. A square cross-section scores 0.71, a circle 1.0, a flat slab near 0.

    An earlier version of this rule had a second "bulky in every direction"
    branch based on bounding-box ratios. Looking at the labels it produced
    showed it firing on flat panels and boxes (a table top scoring 0.14, a
    telephone 0.26) -- shapes with perfectly good faces to pinch. Bounding-box
    ratios mislead on hollow or sparse meshes, so that branch is gone.
    """
    if f["short_mm"] > APERTURE_MM:
        return None, "wider than the gripper opens"
    if f["circularity"] >= 0.70:
        return ("4-finger",
                f"the cross-section is close to circular ({f['circularity']:.2f}), so two "
                f"fingers would let it roll; four fingers cage it")
    return ("2-finger",
            f"the cross-section has flats to pinch against (circularity "
            f"{f['circularity']:.2f}) across a {f['short_mm']:.0f} mm narrow direction")
