#!/usr/bin/env python3
"""Stage 1: density estimation in one dimension.

Needs: fit_independent, independent_pdf
"""

import matplotlib.pyplot as plt
import numpy as np

from _common import banner, guard, notice, section

from anomdet import fit_independent, independent_pdf
from anomdet import data, plotting as P


@guard
def main():
    banner("Stage 1: one feature, one density",
           "Fit p(x) to engine heat measurements, then look at what the model "
           "actually believes. In 1-D you can plot the whole thing, so there "
           "is nowhere for a misunderstanding to hide.")

    d = data.engine_heat_1d()
    x = d.X_train[:, 0]

    section("fit")
    mu, var = fit_independent(d.X_train)
    print(f"  m = {d.X_train.shape[0]} training engines, all of them normal")
    print(f"  fitted   mu = {mu[0]:.3f}   var = {var[0]:.3f}  (sd = {np.sqrt(var[0]):.3f})")
    print(f"  truth    mu = {d.meta['mu_true']:.3f}   sd  = {d.meta['sd_true']:.3f}")

    section("sanity check: does p integrate to 1?")
    # 400-point Gauss-Legendre over +-12 sd is overkill, which is the point:
    # if this is not 1 to ~1e-12, the normalising constant is wrong.
    sd = np.sqrt(var[0])
    xq, wq = np.polynomial.legendre.leggauss(400)
    a, b = mu[0] - 12 * sd, mu[0] + 12 * sd
    xs = 0.5 * (b - a) * xq + 0.5 * (a + b)
    I = 0.5 * (b - a) * np.sum(wq * independent_pdf(xs[:, None], mu, var))
    print(f"  integral of p over [mu-12sd, mu+12sd] = {I:.14f}")

    section("evaluate p at a few specific engines")
    probes = np.array([[70.0], [78.0], [58.0], [92.0]])
    labels = ["dead centre", "+2 sd", "-3 sd", "+5.5 sd"]
    for (xv,), lab, p in zip(probes, labels, independent_pdf(probes, mu, var)):
        print(f"  heat = {xv:5.1f} ({lab:>11s})   p = {p:10.3e}")

    section("thresholding by hand")
    # Deliberately NOT using tune_threshold, that is stage 3. The point
    # here is to feel the trade-off before automating it.
    p_val = independent_pdf(d.X_val, mu, var)
    y = d.y_val.astype(bool)
    print(f"  validation set: {(~y).sum()} normal, {y.sum()} anomalous")
    print(f"\n  {'epsilon':>10}  {'flagged':>8}  {'caught':>8}  {'missed':>7}  {'false alarms':>13}")
    for eps in [1e-2, 3e-3, 1e-3, 3e-4, 1e-5, 1e-8]:
        flag = p_val < eps
        print(f"  {eps:10.0e}  {flag.sum():8d}  {(flag & y).sum():8d}"
              f"  {(~flag & y).sum():7d}  {(flag & ~y).sum():13d}")

    section("figure")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    P.hist_with_normal_fit(ax1, x, mu[0], var[0],
                           title="training data and the fitted density")
    ax1.set_xlabel(d.feature_names[0])
    ax1.set_ylabel("density")

    grid = np.linspace(x.min() - 12, x.max() + 12, 600)
    ax2.semilogy(grid, independent_pdf(grid[:, None], mu, var), color="0.3", lw=1.6,
                 label="p(x)")
    for eps, c in [(1e-3, "#fdae61"), (1e-5, "#d7191c")]:
        ax2.axhline(eps, ls="--", lw=1, color=c, label=f"$\\epsilon = ${eps:g}")
    ax2.scatter(d.X_val[~y, 0], independent_pdf(d.X_val[~y], mu, var), s=12,
                c=P.NORMAL_C, alpha=.6, label="normal")
    ax2.scatter(d.X_val[y, 0], independent_pdf(d.X_val[y], mu, var), s=42,
                facecolors="none", edgecolors=P.ANOM_C, label="anomaly")
    ax2.set_xlabel(d.feature_names[0])
    ax2.set_ylabel("p(x)  (log scale)")
    ax2.set_ylim(1e-14, 1)
    ax2.set_title("the same density, on a log axis", fontsize=9)
    ax2.legend(fontsize=7)
    P.save(fig, "stage1_density_1d.png")

    notice("""
    The left panel is the honest picture of what the model knows, and the
    right panel is the picture you actually make decisions with. On a linear
    density axis the entire anomaly region is pinned indistinguishably to
    zero; on a log axis it spreads over ten orders of magnitude. Every
    thresholding decision lives in that range, which is why the rest of this
    project works in log p.

    Look at the epsilon table. There is no value that catches all ten
    anomalies without also flagging normal engines, and no value that raises
    zero false alarms without missing anomalies. You are choosing a point on
    a trade-off curve, not finding a correct answer. Stage 3 automates the
    choice, but it cannot remove the trade-off.

    The fitted sd is close to the true 4.0 but not equal to it, you have 400
    samples, so expect about 1/sqrt(2m) = 3.5% relative error on sd. Worth
    remembering when you are tempted to read too much into a threshold tuned
    on a few hundred points.
    """)


if __name__ == "__main__":
    main()
