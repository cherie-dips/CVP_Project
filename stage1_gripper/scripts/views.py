"""Render an object mesh into the images the gripper selector sees.

    axis_views(mesh)    three orthographic views along the object's principal axes: down the long
                        axis (the cross-section the label is measured on), then the two side views
    fixed_views(mesh)   two oblique views from fixed directions, like the first fine-tune's renders

Shape only: every object is drawn in the same flat grey, so the selector cannot lean on texture,
and the YCB and Google Scanned objects (scans, no usable textures here) look like the Objaverse
ones. No OpenGL either -- pyrender's pyglet backend does not load on Apple silicon -- so faces are
flat-shaded and painted back to front with PIL at 2x, then downsampled.
"""
import numpy as np, trimesh
from PIL import Image, ImageDraw

PANEL = 256         # px per view
SS = 2              # supersampling
LIGHT = np.array([-0.35, 0.55, 0.76]) / np.linalg.norm([-0.35, 0.55, 0.76])   # in view coordinates


def principal_frame(mesh, n=20000, seed=0):
    """Centre and principal axes (rows: long, middle, short), computed exactly as
    label_geometry.features does, so the end view shows the cross-section it measures."""
    pts = np.asarray(trimesh.sample.sample_surface(mesh, n, seed=seed)[0])
    c = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    p = (pts - c) @ Vt.T
    return c, Vt[np.argsort(-(p.max(0) - p.min(0)))]


def panel(mesh, R, centre, size=PANEL, margin=0.06):
    """One orthographic view. R's rows are the image's right, up, and toward-the-viewer axes."""
    v = (mesh.vertices - centre) @ R.T
    lo, hi = v[:, :2].min(0), v[:, :2].max(0)
    s = (1 - 2 * margin) * size * SS / max(float((hi - lo).max()), 1e-9)
    xy = (v[:, :2] - (lo + hi) / 2) * s + size * SS / 2
    xy[:, 1] = size * SS - xy[:, 1]                                 # image y points down
    shade = 0.40 + 0.55 * np.abs((mesh.face_normals @ R.T) @ LIGHT)  # two-sided: normals are unreliable
    grey = np.clip(205 * shade, 0, 255).astype(np.uint8)
    order = np.argsort(v[mesh.faces, 2].mean(1))                    # farthest first
    polys = xy[mesh.faces[order]].reshape(len(order), 6).tolist()
    img = Image.new("L", (size * SS, size * SS), 255)
    draw = ImageDraw.Draw(img)
    for poly, g in zip(polys, grey[order].tolist()):
        draw.polygon(poly, fill=g)
    return img.resize((size, size), Image.LANCZOS)


def _strip(panels):
    gap = 4
    out = Image.new("L", (len(panels) * PANEL + (len(panels) - 1) * gap, PANEL), 255)
    for i, p in enumerate(panels):
        out.paste(p, (i * (PANEL + gap), 0))
        if i:
            out.paste(225, (i * (PANEL + gap) - gap, 0, i * (PANEL + gap), PANEL))  # thin divider
    return out.convert("RGB")


def axis_views(mesh, seed=0):
    """Left: looking down the long axis. Middle: looking down the short axis. Right: down the middle one."""
    c, (a0, a1, a2) = principal_frame(mesh, seed=seed)
    return _strip([panel(mesh, np.stack([a1, a2, a0]), c),
                   panel(mesh, np.stack([a0, a1, a2]), c),
                   panel(mesh, np.stack([a0, a2, a1]), c)])


def fixed_views(mesh, up=(0, 1, 0), views=((35, 25), (125, 25))):
    """Two oblique views (azimuth, elevation in degrees) in the mesh's own frame, y up as in glTF."""
    up = np.asarray(up, float)
    c = mesh.bounds.mean(0)
    panels = []
    for az, el in views:
        az, el = np.radians(az), np.radians(el)
        toward = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
        if abs(up[2]) > 0.5:                                         # z-up meshes (YCB, USD)
            toward = toward[[0, 2, 1]] * [1, -1, 1]
        right = np.cross(up, toward); right /= np.linalg.norm(right)
        panels.append(panel(mesh, np.stack([right, np.cross(toward, right), toward]), c))
    return _strip(panels)
