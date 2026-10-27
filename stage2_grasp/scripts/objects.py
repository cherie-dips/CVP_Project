"""Object meshes in metres, from the three sources the project uses.

    objaverse(uid)   data/objaverse/meshes/<uid>.glb, times GraspGen's real-world scale    (y up)
    ycb(name)        data/ycb/<name>/google_16k/nontextured.ply                              (z up)
    usd(path)        an Isaac Sim asset in selected_objects/ (YCB and Google Scanned Objects)  (z up)
"""
import json, pathlib

import numpy as np, trimesh

ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCALES = None


def objaverse(uid):
    global _SCALES
    if _SCALES is None:
        _SCALES = json.loads((ROOT/"data"/"objaverse"/"scales.json").read_text())
    mesh = trimesh.load(ROOT/"data"/"objaverse"/"meshes"/f"{uid}.glb", force="mesh")
    mesh.apply_scale(_SCALES[uid])
    return mesh


def ycb(name):
    return trimesh.load(ROOT/"data"/"ycb"/name/"google_16k"/"nontextured.ply", force="mesh")


def usd(path):
    """The visual mesh of an Isaac Sim asset, in metres, with its prim transforms applied."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(path))
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    xform = UsdGeom.XformCache()
    parts = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputePurpose() != UsdGeom.Tokens.default_:
            continue                                    # skip the collision copy (purpose=guide)
        m = UsdGeom.Mesh(prim)
        v = np.asarray(m.GetPointsAttr().Get(), float)
        T = np.asarray(xform.GetLocalToWorldTransform(prim))   # row-vector convention
        v = (v @ T[:3, :3] + T[3, :3]) * scale
        counts = np.asarray(m.GetFaceVertexCountsAttr().Get())
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get())
        faces, k = [], 0
        for c in counts:                                # fan-triangulate any n-gons
            faces += [[idx[k], idx[k+j], idx[k+j+1]] for j in range(1, c-1)]
            k += c
        parts.append(trimesh.Trimesh(v, np.array(faces), process=False))
    return trimesh.util.concatenate(parts)
