#!/usr/bin/env python3
"""Which gripper, and how hard to squeeze — the physics layer.

GraspGenX answers WHERE to grasp (it takes a point cloud, so it cannot see mass, friction or
fragility). This module answers WHICH GRIPPER and HOW MUCH FORCE, from estimated object
properties. See presentation-plan.md section 9.

Two constraints:
    slip     n * mu * F  >=  s * m * (g + a)
    damage   F / A_contact  <=  sigma_max

and the part that actually decides it: real grippers are POSITION controlled, so the force you
get is stiffness x closure overshoot, F = k_eff * delta, whether you wanted it or not. A rigid
pad on a rigid object turns 1.5 mm of error into ~250 N. That is why compliance matters.

    python force_model.py                 # illustrative table
    python force_model.py --delta 0.0005  # what a better-calibrated gripper buys
"""
import argparse
from dataclasses import dataclass

G = 9.81


@dataclass
class Gripper:
    name: str
    n_contacts: int
    contact_area: float   # m^2 per finger — MEASURE with carbon paper
    stiffness: float      # N/m — MEASURE by pressing the finger onto a kitchen scale
    min_aperture: float = 0.0
    max_aperture: float = 1.0


@dataclass
class Obj:
    name: str
    mass: float           # kg   — VLM / image2mass
    mu: float             # -    — VLM material ID -> friction table
    sigma_max: float      # Pa   — VLM + literature. WEAKEST LINK.
    k_object: float       # N/m  — rigid / soft / thin-shell prior


@dataclass
class Decision:
    gripper: str
    f_slip: float         # force needed not to drop it
    f_applied: float      # force actually delivered (>= f_slip, set by overshoot)
    stress: float         # Pa at the contact
    safe: bool
    margin: float         # sigma_max / stress; >1 is safe


def evaluate(obj: Obj, grip: Gripper, delta: float, safety: float, a_max: float) -> Decision:
    f_slip = safety * obj.mass * (G + a_max) / (grip.n_contacts * obj.mu)
    k_eff = 1.0 / (1.0 / grip.stiffness + 1.0 / obj.k_object)
    f_applied = max(f_slip, k_eff * delta)
    stress = f_applied / grip.contact_area
    return Decision(grip.name, f_slip, f_applied, stress,
                    stress <= obj.sigma_max, obj.sigma_max / stress)


def select(obj, grippers, delta, safety, a_max, feasible=None):
    """Lexicographic: geometric feasibility (GraspGenX) -> safety -> prefer the simpler gripper.

    `feasible` maps gripper name -> bool, supplied by GraspGenX. Defaults to all-feasible, but
    it is what lets the 2-finger win on thin/flat/cluttered objects the 4-finger cannot reach.
    """
    results = [evaluate(obj, g, delta, safety, a_max) for g in grippers]
    ok = [(g, d) for g, d in zip(grippers, results)
          if d.safe and (feasible is None or feasible.get(g.name, True))]
    if not ok:
        return None, results          # needs true force control, or a different gripper
    return ok[0][1], results          # grippers passed in preference order (simplest first)


# Illustrative only — every value here is an estimate, not a measurement.
GRIPPERS = [
    Gripper("2f rigid pad", 2, 1.5e-4, 2.0e5),
    Gripper("4f fin-ray",   4, 4.0e-4, 3.0e3),
]
OBJECTS = [
    Obj("egg",               0.060, 0.45, 1.2e6, 1.0e6),
    Obj("tomato",            0.120, 0.50, 0.15e6, 3.0e3),
    Obj("water bottle full", 0.550, 0.55, 0.8e6, 2.0e4),
    Obj("water bottle empty",0.020, 0.55, 0.3e6, 5.0e3),
    Obj("glass cup",         0.250, 0.40, 5.0e6, 5.0e6),
    Obj("oiled steel ball",  0.450, 0.12, 50e6,  1.0e7),
    Obj("metal block",       2.000, 0.60, 50e6,  1.0e7),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.0015, help="closure overshoot, m — MEASURE IT")
    ap.add_argument("--safety", type=float, default=1.5)
    ap.add_argument("--a_max", type=float, default=2.0, help="max arm acceleration, m/s^2")
    args = ap.parse_args()

    print(f"closure overshoot delta = {args.delta * 1000:.2f} mm, safety = {args.safety}\n")
    print(f"{'object':>20} | {'gripper':>13} {'F_slip':>7} {'F_app':>8} {'MPa':>7} "
          f"{'limit':>7} {'margin':>7}")
    print("-" * 92)
    for obj in OBJECTS:
        chosen, results = select(obj, GRIPPERS, args.delta, args.safety, args.a_max)
        for d in results:
            tag = "  <-- USE" if chosen and d.gripper == chosen.gripper else (
                  "  damages" if not d.safe else "")
            print(f"{obj.name:>20} | {d.gripper:>13} {d.f_slip:>7.2f} {d.f_applied:>8.2f} "
                  f"{d.stress / 1e6:>7.3f} {obj.sigma_max / 1e6:>7.2f} {d.margin:>7.1f}{tag}")
        if chosen is None:
            print(f"{'':>20} | NEITHER is safe — needs true force control")
        print()


if __name__ == "__main__":
    main()
