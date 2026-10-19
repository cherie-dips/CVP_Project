# Fin-ray gripper — description

This is the file the pipeline hands to the LLM alongside the object image. Every
number is measured from the STLs in this folder (see `setup/derive_gripper_params.py`).

## One gripper, two configurations

Four identical fin-ray fingers sit in a cross around a central axis, each held in a
clamp that pivots on the body. A toothed rack down the axis drives all four clamps
together, so the fingers open and close as one.

**Removing two opposing fingers turns it into the 2-finger gripper.** Nothing else
changes — same body, same clamps, same fingers, same travel. The only difference is
how many fingers close on the object.

| | contacts | closes along |
|---|---|---|
| **2-finger** | 2 | one axis |
| **4-finger** | 4 | two axes, caging the object |

## Finger

| | mm |
|---|---|
| length | 85 |
| width (in the radial plane, tapering to a point) | 30 |
| thickness (the contact strip) | 10 |
| material | TPU, fin-ray rib structure — it conforms to whatever it is pressed against |

## Mechanism

| | mm |
|---|---|
| pivot radius from the axis | 30.5 |
| pivot height, below the body's top plate | 6.2 |
| clamp plate thickness / clevis gap | 13.0 / 14.0 |
| fingertip separation, fully open | 139 |

Opposing fingers meet at a clamp angle of −23.3°; the travel beyond that is set by
the rack stroke.

## Choosing between them

A 2-finger grip needs two opposing flats to pinch against. Round cross-sections have
none — the object rolls out of a two-point pinch — and that is what the second pair
of fingers is for. Long, prismatic or flat-sided objects are the 2-finger case;
round, squat or irregular ones are the 4-finger case.
