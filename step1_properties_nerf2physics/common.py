"""Shared helpers for Step 1: material densities and the MnRE metric."""
import numpy as np

DENSITY = {
    "steel": 7850, "stainless": 7850, "iron": 7870, "metal": 7800, "brass": 8500,
    "copper": 8960, "aluminum": 2700, "aluminium": 2700, "zinc": 7140,
    "glass": 2500, "ceramic": 2300, "porcelain": 2400, "stone": 2600, "marble": 2700,
    "concrete": 2400, "rubber": 1200, "silicone": 1200, "silicon": 1200,
    "acrylic": 1180, "pvc": 1400, "abs": 1050, "nylon": 1140, "polyester": 1380,
    "polypropylene": 900, "polyethylene": 950, "plastic": 950, "resin": 1200,
    "wood": 600, "bamboo": 700, "oak": 750, "pine": 500, "mdf": 750, "plywood": 600,
    "leather": 900, "cotton": 400, "canvas": 400, "linen": 400, "wool": 300,
    "fabric": 350, "foam": 60, "paper": 800, "cardboard": 700,
}
DEFAULT_DENSITY = 800.0


def density_of(material):
    if not isinstance(material, str):
        return np.nan
    m = material.lower()
    for k, v in DENSITY.items():           # longest key first so "stainless steel" wins
        pass
    for k in sorted(DENSITY, key=len, reverse=True):
        if k in m:
            return DENSITY[k]
    return np.nan


def mnre(pred, true):
    """Mean normalised relative error, the NeRF2Physics metric. 1.0 is perfect."""
    pred = np.clip(np.asarray(pred, float), 1e-9, None)
    true = np.clip(np.asarray(true, float), 1e-9, None)
    return float(np.mean(np.minimum(pred / true, true / pred)))


