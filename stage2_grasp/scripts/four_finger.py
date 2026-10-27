"""4-finger grasps from GraspGenX, by composing two 2-finger pinches.

GraspGenX's released checkpoint knows a gripper only through 12 numbers, the open and half-open
swept boxes (gripper_backbone = sweep_volume_v2; the gripper type is not an input). Every gripper
it was trained on closes along one axis, so the box narrows in x between open and half-open and
keeps its y. Our 4-finger box narrows in both: no training gripper looks like that, and the model
reads it as a bulky hand and backs off.

But the 4-finger gripper *is* two of our 2-finger grippers sharing an approach axis, the second
turned 90 degrees, both closing together. So a 4-finger grasp at pose T is good when the 2-finger
grasp at T holds and so does the 2-finger grasp at T turned 90 degrees about its approach axis --
and GraspGenX's own discriminator, conditioned on the 2-finger box it does understand, can score
both.

    candidates   GraspGenX's own planner with the 2-finger gripper: diffusion samples plus its
                 oriented-bounding-box sweep (GraspMoE), so approaches along the object's axes,
                 where four fingers can cage it, are always among them
    yaw search   each candidate turned 0, 15, ..., 75 degrees about its approach axis (with four
                 fingers in a cross, 90 degrees brings back the same grasp)
    score        min(D(T Rz(yaw)), D(T Rz(yaw + 90))): the weaker pair decides
    one rack     what the discriminator cannot know: both pairs are driven together, so the pair
                 that grips first is pushed on until the other grips, and the fin-ray can bend
                 only so far (10 mm, finray.DEFLECT_MM). So the object must be about as wide under
                 one pair as under the other -- within 2 x 10 mm, measured from the point cloud in
                 the slab each pair closes through -- or the grasp is dropped. Without this the
                 planner happily pinches a can across its diameter with one pair and end to end
                 with the other.

extra_candidates adds poses from any other source (evaluate_grasps.py passes GraspGenX's grasps for
the square 4-finger box measured from the STLs); they go through the same scoring.
"""
import numpy as np, torch

from graspgenx.samplers.graspmoe import _score_grasps_with_discriminator, run_graspmoe

YAWS = np.radians(np.arange(0, 90, 15))
BAND = (0.053, 0.138)        # m above the mounting face: where the real fingers are, open (finray.py)
SLAB = 0.010                 # m either side of a pair's closing plane: the 10 mm finger, plus the patch it presses
MAX_MISMATCH = 0.020         # m: 2 x finray.DEFLECT_MM


def rz(a):
    c, s = np.cos(a), np.sin(a)
    R = np.eye(4)
    R[:2, :2] = [[c, -s], [s, c]]
    return R


def discriminator(pts, sampler, poses, batch=4096):
    """GraspGenX's discriminator on given poses (object frame), with the sampler's gripper box."""
    centre = pts.mean(0)
    pc = torch.from_numpy((pts - centre).astype(np.float32))
    out = [_score_grasps_with_discriminator(poses[i:i + batch].astype(np.float32), pc, centre, sampler)
           for i in range(0, len(poses), batch)]
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def candidates(pts, sampler_2f, num_grasps=200):
    """GraspMoE with the 2-finger gripper: diffusion samples and the OBB sweep, always both."""
    # the OBB sweep's own yaw steps would repeat the yaw search below, so only 0 and 180 degrees
    moe = run_graspmoe(pts, sampler_2f, num_grasps=num_grasps, topk_num_grasps=-1, num_yaws=2,
                       obb_density="dense-topandside", skip_obb_rule="never")
    poses = np.concatenate([moe["grasps_diff"], moe["grasps_obb"]])
    source = np.array(["diffusion"] * len(moe["grasps_diff"]) + ["obb"] * len(moe["grasps_obb"]))
    return poses, source


def widths(pts, poses):
    """Object width under each pair: along x within the slab |y| < SLAB, and along y within
    |x| < SLAB, both in the finger band. NaN where a pair would close on nothing."""
    out = np.full((len(poses), 2), np.nan)
    for i, T in enumerate(poses):
        q = (pts - T[:3, 3]) @ T[:3, :3]                     # object in the gripper frame
        q = q[(q[:, 2] > BAND[0]) & (q[:, 2] < BAND[1])]
        for k, (along, across) in enumerate(((0, 1), (1, 0))):
            sl = q[np.abs(q[:, across]) < SLAB, along]
            if len(sl) >= 5:
                out[i, k] = sl.max() - sl.min()
    return out


def plan(pts, sampler_2f, topk=None, num_grasps=200, extra_candidates=None, one_rack=True):
    """4-finger grasps, best first: (poses, score, pair scores (n, 2), candidate source)."""
    cands, source = candidates(pts, sampler_2f, num_grasps)
    if extra_candidates is not None and len(extra_candidates):
        cands = np.concatenate([cands, extra_candidates])
        source = np.concatenate([source, ["extra"] * len(extra_candidates)])
    A = np.einsum("nij,yjk->nyik", cands, np.stack([rz(y) for y in YAWS]))
    B = np.einsum("nij,yjk->nyik", cands, np.stack([rz(y + np.pi / 2) for y in YAWS]))
    sa = discriminator(pts, sampler_2f, A.reshape(-1, 4, 4)).reshape(len(cands), len(YAWS))
    sb = discriminator(pts, sampler_2f, B.reshape(-1, 4, 4)).reshape(len(cands), len(YAWS))
    s4 = np.minimum(sa, sb)
    if one_rack:
        w = widths(pts, A.reshape(-1, 4, 4)).reshape(len(cands), len(YAWS), 2)
        ok = np.abs(w[..., 0] - w[..., 1]) <= MAX_MISMATCH          # NaN (a pair touches nothing) -> False
        s4 = np.where(ok, s4, -1.0 + s4)                             # infeasible ranks below every feasible one
    best = s4.argmax(1)
    idx = np.arange(len(cands))
    order = np.argsort(-s4[idx, best])[:topk]
    return (A[idx, best][order], s4[idx, best][order],
            np.stack([sa[idx, best], sb[idx, best]], 1)[order], source[order])
