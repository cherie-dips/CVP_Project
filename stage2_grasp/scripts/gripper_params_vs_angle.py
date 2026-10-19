"""The 12 GraspGenX swept-volume values as a function of how far the gripper opens.

Everything is measured off the assembled STLs (see scripts/assemble_gripper.py).
The only free parameter is the opening angle, which is set by the rack stroke --
and the rack is not among the supplied STL files.
"""
import pathlib, sys, json, numpy as np, trimesh, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, str(pathlib.Path(__file__).parent))
from assemble_gripper import FING, finger_in_clamp, clamp_to_body

Z_BASE = -49.53        # body mounting face = gripper base link
SIGN   = +1            # finger outboard of the pivot (confirmed from the video)

def pts(theta, about=0.0, n=140000):
    M = clamp_to_body(SIGN, theta, about)
    f = FING.copy(); f.apply_transform(M @ finger_in_clamp())
    return np.asarray(trimesh.sample.sample_surface(f, n)[0])

def tip_gap(theta):
    p = pts(theta); up = p[p[:,2] > p[:,2].max()*0.5]
    return 2*up[:,0].min()

def box(theta, n_fingers):
    # removing two of the four fingers leaves an OPPOSING pair: 0 and 180 deg
    step = 90 if n_fingers == 4 else 180
    per = [pts(theta, step*k) for k in range(n_fingers)]
    z_hi = min(p[:,2].max() for p in per); z_lo = max(p[:,2].min() for p in per)
    band = [p[(p[:,2] >= z_lo) & (p[:,2] <= z_hi)] for p in per]
    ex = 2*band[0][:,0].min()
    ey = 2*band[1][:,1].min() if n_fingers == 4 else 2*np.abs(band[0][:,1]).max()
    return ex, ey, z_hi - z_lo, (z_hi + z_lo)/2 - Z_BASE

# angle at which opposing fingers touch
lo, hi = -60.0, 60.0
for _ in range(40):
    mid = (lo+hi)/2
    if tip_gap(mid) > 0: hi = mid
    else: lo = mid
TH_CLOSED = hi
print(f"opposing fingers touch at theta = {TH_CLOSED:.1f} deg  (fully closed)\n")
print(f"{'open':>5} {'sweep':>6} {'tip gap':>8} | {'ex':>6} {'ey(4f)':>7} {'ey(2f)':>7} "
      f"{'ez':>6} {'oz':>6} | half-open {'ex':>6} {'ez':>6} {'oz':>6}   (mm)")

table = {}
for th_open in [30, 40, 50, 60, 70, 80]:
    th_mid = (th_open + TH_CLOSED)/2
    o4 = box(th_open, 4); m4 = box(th_mid, 4)
    o2 = box(th_open, 2); m2 = box(th_mid, 2)
    print(f"{th_open:5.0f} {th_open-TH_CLOSED:6.1f} {tip_gap(th_open):8.1f} | "
          f"{o4[0]:6.1f} {o4[1]:7.1f} {o2[1]:7.1f} {o4[2]:6.1f} {o4[3]:6.1f} | "
          f"{'':9s} {m4[0]:6.1f} {m4[2]:6.1f} {m4[3]:6.1f}")
    mm = lambda v: round(v/1000, 4)
    table[str(th_open)] = {
      "2-finger": {"extents_open": [mm(o2[0]), mm(o2[1]), mm(o2[2])],
                   "offset_open": [0.0, 0.0, mm(o2[3])],
                   "extents_mid": [mm(m2[0]), mm(m2[1]), mm(m2[2])],
                   "offset_mid": [0.0, 0.0, mm(m2[3])], "n_contacts": 2},
      "4-finger": {"extents_open": [mm(o4[0]), mm(o4[1]), mm(o4[2])],
                   "offset_open": [0.0, 0.0, mm(o4[3])],
                   "extents_mid": [mm(m4[0]), mm(m4[1]), mm(m4[2])],
                   "offset_mid": [0.0, 0.0, mm(m4[3])], "n_contacts": 4}}

json.dump({"theta_closed_deg": round(TH_CLOSED, 2), "by_open_angle_deg": table},
          open("data/gripper/params_by_angle.json", "w"), indent=1)
print("\nsaved data/gripper/params_by_angle.json")
