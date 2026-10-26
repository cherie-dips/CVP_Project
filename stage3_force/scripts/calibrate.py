"""SEN0704 calibration: known weights on the pad vs the sum of the 6x6 grid.

Fits  sum = a*F + b  by least squares, dropping points more than 2.5 robust
standard deviations off the line, and inverts it for counts_to_newton().

    python stage3_force/scripts/calibrate.py      # -> results/calibration.png
"""
import os

import numpy as np

G = 9.81
# (grams on the pad, sum of the 36 cells)
MEASURED = np.array([
    (48.2, 252), (100.6, 244), (149.8, 466), (197.8, 796), (246.6, 1020), (292.4, 1257),
    (360.2, 1450), (191.8, 734), (484.0, 1490), (40.0, 17), (344.6, 1574), (368.2, 1679),
    (390.4, 1746), (458.0, 2003), (840.0, 3752), (1713.4, 7901), (873.0, 3976),
])


def fit():
    F, s = MEASURED[:, 0] * G / 1000, MEASURED[:, 1]
    keep = np.ones(len(F), bool)
    for _ in range(10):
        a, b = np.polyfit(F[keep], s[keep], 1)
        r = s - (a * F + b)
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        new = np.abs(r) < 2.5 * mad
        if (new == keep).all():
            break
        keep = new
    r2 = 1 - (r[keep] ** 2).sum() / ((s[keep] - s[keep].mean()) ** 2).sum()
    return a, b, keep, r2


A, B, KEEP, R2 = fit()


def counts_to_newton(total):
    """Grid sum -> Newtons. An all-zero grid is 0 N, not the line's intercept."""
    total = np.asarray(total, float)
    return np.where(total > 0, np.clip((total - B) / A, 0, None), 0.0)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F, s = MEASURED[:, 0] * G / 1000, MEASURED[:, 1]
    err = (s[KEEP] - B) / A - F[KEEP]
    print(f"sum = {A:.1f}*F {B:+.1f}   R2 = {R2:.4f}   ({KEEP.sum()} of {len(F)} points)")
    print(f"F[N] = {1 / A:.5f}*sum {-B / A:+.3f}    RMSE {np.sqrt((err ** 2).mean()):.3f} N")
    print("dropped:", ", ".join(f"{g:g} g -> {c:g}" for g, c in MEASURED[~KEEP]))

    x = np.linspace(0, F.max() * 1.05, 50)
    plt.figure(figsize=(6, 4.5))
    plt.scatter(F[KEEP], s[KEEP], label="used")
    plt.scatter(F[~KEEP], s[~KEEP], marker="x", c="r", s=60, label="dropped")
    plt.plot(x, A * x + B, "k-", label=f"sum = {A:.0f}·F {B:+.0f}  (R² = {R2:.3f})")
    plt.xlabel("Force (N)"); plt.ylabel("Sum of the 6×6 grid"); plt.title("SEN0704 calibration")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "results", "calibration.png")
    plt.savefig(out, dpi=130)
    print("->", os.path.abspath(out))
