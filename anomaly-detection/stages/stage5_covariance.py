#!/usr/bin/env python3
"""Stage 5: dropping the independence assumption.

Needs: fit_independent, log_independent_pdf, fit_correlated,
       log_correlated_pdf, tune_threshold_log, precision_recall_f1
"""

import matplotlib.pyplot as plt
import numpy as np

from _common import banner, guard, notice, section

from anomdet import (fit_independent, log_independent_pdf, fit_correlated,
                     log_correlated_pdf, tune_threshold_log,
                     precision_recall_f1)
from anomdet import data, plotting as P


@guard
def main():
    banner("Stage 5: the full covariance model",
           "A dataset where the axis-aligned model is not slightly worse but "
           "essentially blind: every anomaly is within 2.1 sd of the mean in "
           "EVERY coordinate, while being jointly impossible.")

    d = data.correlated_data()
    y_val, y_test = d.y_val, d.y_test
    anom = y_test.astype(bool)

    section("the anomalies, one coordinate at a time")
    mu_d, var_d = fit_independent(d.X_train)
    sd = np.sqrt(var_d)
    z = np.abs(d.X_test - mu_d) / sd
    print(f"  correlation in the training data: {np.corrcoef(d.X_train, rowvar=False)[0,1]:+.3f}")
    print(f"  worst per-coordinate |z| over the {anom.sum()} test anomalies: "
          f"{z[anom].max():.2f}")
    print(f"  ... and over the {(~anom).sum()} normal test points:            "
          f"{z[~anom].max():.2f}")
    print("\n  So marginally, the anomalies are LESS extreme than the most extreme")
    print("  normal points. No per-feature rule, no z-score cut, no box, no")
    print("  product of marginals, can separate them. The information is")
    print("  entirely in the joint structure.")

    section("Mahalanobis distance, which does see it")
    mu_f, Sigma = fit_correlated(d.X_train)
    L = np.linalg.cholesky(Sigma)
    dist = np.sqrt(np.sum(np.linalg.solve(L, (d.X_test - mu_f).T) ** 2, axis=0))
    print(f"  d over anomalies: [{dist[anom].min():.2f}, {dist[anom].max():.2f}]")
    print(f"  d over normals:   [{dist[~anom].min():.2f}, {dist[~anom].max():.2f}]")
    print("\n  fitted Sigma =")
    for row in Sigma:
        print("      [" + "  ".join(f"{v:8.4f}" for v in row) + " ]")
    ev = np.linalg.eigvalsh(Sigma)
    print(f"  eigenvalues {ev[0]:.4f}, {ev[1]:.4f}  -> variance ratio "
          f"{ev[1]/ev[0]:.1f}, so the cloud is")
    print(f"  sqrt of that = {np.sqrt(ev[1]/ev[0]):.1f} times longer than it is wide. The")
    print("  diagonal model has to cover that sliver with a nearly circular")
    print("  blob, so it spends probability mass exactly where the anomalies")
    print("  live and starves the ends of the ridge where normal points are.")

    section("head to head")
    rows = []
    for name, logp in (("diagonal  (independent)",
                        lambda X: log_independent_pdf(X, mu_d, var_d)),
                       ("full covariance",
                        lambda X: log_correlated_pdf(X, mu_f, Sigma))):
        t, f1v = tune_threshold_log(y_val, logp(d.X_val), n_steps=4000)
        pred = logp(d.X_test) < t
        pr, rc, f1 = precision_recall_f1(y_test, pred)
        rows.append((name, f1v, pr, rc, f1, pred, logp))
    print(f"  {'model':<26}{'val F1':>8}{'precision':>11}{'recall':>8}{'test F1':>9}")
    for name, f1v, pr, rc, f1, _, _ in rows:
        print(f"  {name:<26}{f1v:8.3f}{pr:11.3f}{rc:8.3f}{f1:9.3f}")

    section("what it costs you")
    n = d.X_train.shape[1]
    print(f"  parameters: diagonal 2n = {2*n},  full n + n(n+1)/2 = {n + n*(n+1)//2}")
    print("  The full model needs O(n^2) parameters, so it needs more data: with")
    print("  m < n the sample covariance is singular and the Cholesky factor")
    print("  does not exist. Rules of thumb: m > 10n before you trust it, and if")
    print("  you must go further, regularise with Sigma + lambda*I (shrinkage).")
    print("  The diagonal model has the opposite profile: it is cheap, it")
    print("  works at m ~ n, and you can always recover correlations by hand:")
    print("  add x1/x2 or x1*x2 as an extra feature and the product model can")
    print("  suddenly express the ratio. That trick is what Ng's lecture means")
    print("  by engineering combination features.")

    section("figure")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
    for ax, (name, f1v, pr, rc, f1, pred, logp) in zip(axes, rows):
        P.scatter_labelled(ax, d.X_test, y=y_test, flagged=pred,
                           feature_names=d.feature_names)
        P.density_contours(ax, logp, (-3.2, 3.2), (-3.2, 3.2),
                           decades=(1, 2, 4, 7, 11))
        ax.set_title(f"{name}\ntest F1 = {f1:.3f}   "
                     f"(P {pr:.2f} / R {rc:.2f})", fontsize=9)
        ax.set_aspect("equal")
    P.save(fig, "stage5_covariance.png")

    notice("""
    The two panels use identical data and identical machinery; the only
    difference is whether Sigma is allowed off-diagonal entries. The diagonal
    contours are nearly circles and they swallow the anomalies whole, so no
    threshold can isolate them, the flagged crosses land on normal points
    out at the edges instead. The full-covariance contours follow the ridge
    and the anomalies fall outside them.

    This is the geometric content of the independence assumption. A product
    of marginals can only produce axis-aligned level sets, and 'axis-aligned'
    is a statement about the coordinate system you happened to write the data
    down in, which is not a property of the physics. The full model is
    invariant under rotation of the features; the diagonal one is not. The
    test suite checks that invariance explicitly.

    Worth doing: rotate this dataset by 45 degrees before fitting (X @ Q.T
    with Q a rotation). The full-covariance F1 will not move at all. The
    diagonal F1 will, because at 45 degrees the correlated cloud happens to
    align better with the axes. If a model's score depends on the angle you
    chose to view the data from, that dependence is a bug in the model, not
    a feature of the data.
    """)


if __name__ == "__main__":
    main()
