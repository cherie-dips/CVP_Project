import pathlib, sys, json, numpy as np, trimesh, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, str(pathlib.Path(__file__).parent))
from assemble_gripper import FING, finger_in_clamp, clamp_to_body

def finger_pts(sign, theta, about=0.0, n=200000):
    M = clamp_to_body(sign, theta, about)
    f = FING.copy(); f.apply_transform(M @ finger_in_clamp())
    return np.asarray(trimesh.sample.sample_surface(f, n)[0])

def aperture(sign, theta):
    p = finger_pts(sign, theta)
    up = p[p[:,2] > p[:,2].max()*0.5]
    return 2*up[:,0].min()

# find the angle at which opposing fingers touch
for sign, name in [(+1, "outward"), (-1, "inward")]:
    lo, hi = -60.0, 60.0
    for _ in range(40):
        mid = (lo+hi)/2
        if aperture(sign, mid) > 0: hi = mid
        else: lo = mid
    close = hi
    print(f"{name:8s} mounting: fingers meet at theta = {close:6.2f} deg, "
          f"aperture at +70 deg from there = {aperture(sign, close+70):6.1f} mm")

SIGN = +1
lo, hi = -60.0, 60.0
for _ in range(40):
    mid = (lo+hi)/2
    if aperture(SIGN, mid) > 0: hi = mid
    else: lo = mid
TH_CLOSED = hi
TH_OPEN   = TH_CLOSED + 70.0
TH_MID    = TH_CLOSED + 35.0
print(f"\nclosed {TH_CLOSED:.1f} deg -> open {TH_OPEN:.1f} deg (70 deg sweep), "
      f"half-open {TH_MID:.1f} deg")

Z_BASE = -49.53      # body mounting face: GraspGenX measures from the gripper base

def box(theta, n_fingers):
    per = [finger_pts(SIGN, theta, 90*k) for k in range(n_fingers)]
    z_hi = min(p[:,2].max() for p in per)
    z_lo = max(p[:,2].min() for p in per)
    band = [p[(p[:,2] >= z_lo) & (p[:,2] <= z_hi)] for p in per]
    x_half = min(b[:,0].min() for b in [band[0]])                      # +x finger
    if n_fingers == 4:
        y_half = min(b[:,1].min() for b in [band[1]])                  # +y finger
    else:
        y_half = np.abs(band[0][:,1]).max()                            # finger breadth
    return dict(ex=2*x_half, ey=2*y_half, ez=z_hi-z_lo,
                oz=(z_hi+z_lo)/2 - Z_BASE)

out = {}
for n, tag in [(2, "2-finger"), (4, "4-finger")]:
    o, m = box(TH_OPEN, n), box(TH_MID, n)
    out[tag] = {"extents_open": [round(o["ex"]/1000,4), round(o["ey"]/1000,4), round(o["ez"]/1000,4)],
                "offset_open":  [0.0, 0.0, round(o["oz"]/1000,4)],
                "extents_mid":  [round(m["ex"]/1000,4), round(m["ey"]/1000,4), round(m["ez"]/1000,4)],
                "offset_mid":   [0.0, 0.0, round(m["oz"]/1000,4)],
                "n_contacts": n}
    print(f"\n{tag}:  open box {o['ex']:6.1f} x {o['ey']:6.1f} x {o['ez']:6.1f} mm @ z {o['oz']:6.1f}")
    print(f"{'':10s} mid  box {m['ex']:6.1f} x {m['ey']:6.1f} x {m['ez']:6.1f} mm @ z {m['oz']:6.1f}")

json.dump({"source": "measured from gripper/*.stl",
           "pivot_radius_mm": 30.50, "pivot_z_mm": -6.19,
           "theta_closed_deg": round(TH_CLOSED,2), "theta_open_deg": round(TH_OPEN,2),
           "grippers": out}, open("gripper/gripper_params_measured_raw.json","w"), indent=1)
