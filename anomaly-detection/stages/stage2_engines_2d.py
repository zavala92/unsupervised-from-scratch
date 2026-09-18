#!/usr/bin/env python3
"""Stage 2: two features, and the shape the independence assumption forces.

Needs: fit_independent, independent_pdf, log_independent_pdf
"""

import matplotlib.pyplot as plt
import numpy as np

from _common import banner, guard, notice, section

from anomdet import fit_independent, independent_pdf, log_independent_pdf
from anomdet import data, plotting as P


@guard
def main():
    banner("Stage 2: heat and vibration",
           "The lecture's aircraft-engine picture, built for real. Two "
           "features, contours of p, and a first look at the price of "
           "modelling the features as independent.")

    d = data.engine_data()
    mu, var = fit_independent(d.X_train)

    section("fit")
    for j, name in enumerate(d.feature_names):
        print(f"  {name:20s} mu = {mu[j]:7.3f}   sd = {np.sqrt(var[j]):6.3f}")

    section("three engines off the line")
    # Probes 2 and 3 are each about 2 sd from the mean in BOTH coordinates.
    # The only difference is the sign: probe 2 follows the heat-vibration
    # pattern the fleet obeys, probe 3 runs against it.
    probes = np.array([[70.0, 10.0], [76.0, 12.5], [77.3, 7.0]])
    tags = ["textbook normal", "warm and rough", "warm and smooth"]

    sd = np.sqrt(var)
    # A preview of stage 5, computed here with plain numpy so you can compare
    # the independent model's verdict against a model that knows about the
    # correlation.  Mahalanobis distance d: normal data has d of order 1-3.
    Sig = np.cov(d.X_train, rowvar=False)
    L = np.linalg.cholesky(Sig)
    dist = np.sqrt(np.sum(np.linalg.solve(L, (probes - mu).T) ** 2, axis=0))

    print(f"  {'engine':>16}  {'z(heat)':>8}{'z(vib)':>8}   {'p (indep)':>11}   {'d (full cov)':>12}")
    for (h, v), tag, pv, dd in zip(probes, tags, independent_pdf(probes, mu, var), dist):
        z = ((np.array([h, v]) - mu) / sd)
        print(f"  {tag:>16}  {z[0]:+8.2f}{z[1]:+8.2f}   {pv:11.3e}   {dd:12.2f}")

    section("the independence assumption, made visible")
    # The data were generated with correlated heat and vibration. The
    # independent model cannot represent that, so compare what it believes
    # against what the data say.
    corr_data = np.corrcoef(d.X_train, rowvar=False)[0, 1]
    print(f"  correlation between the two features in the data:  {corr_data:+.3f}")
    print(f"  correlation the independent model can represent:   {0.0:+.3f}")
    print("\n  Consequence: the model's contours are axis-aligned ellipses, while the")
    print("  data cloud is a tilted one. Look back at the probe table: the")
    print("  independent model scores engines 2 and 3 within a factor of ~3 of")
    print("  each other, because both are ~2 sd out in each coordinate and a")
    print("  product of marginals cannot tell the two signs apart. A model that")
    print("  knows the correlation separates them cleanly, d = 4.0 versus 2.0,")
    print("  which is the difference between 'inspect this' and 'ship it'.")

    section("log p, and why you want it")
    logp = log_independent_pdf(d.X_val, mu, var)
    pv = independent_pdf(d.X_val, mu, var)
    y = d.y_val.astype(bool)
    print(f"  normal   examples: log p in [{logp[~y].min():8.2f}, {logp[~y].max():8.2f}]")
    print(f"  anomalous examples: log p in [{logp[y].min():8.2f}, {logp[y].max():8.2f}]")
    print(f"  smallest p on the validation set: {pv.min():.3e}")
    print("  Still comfortably representable at n = 2. Stage 4 breaks it.")

    section("figure")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    P.scatter_labelled(axes[0], d.X_train, feature_names=d.feature_names)
    P.density_contours(axes[0], lambda X: log_independent_pdf(X, mu, var),
                       (48, 96), (2, 19))
    axes[0].scatter(probes[:, 0], probes[:, 1], marker="*", s=200, c="k",
                    zorder=5)
    for (h, v), lab in zip(probes, "123"):
        axes[0].annotate(lab, (h, v), textcoords="offset points",
                         xytext=(8, 6), fontsize=9, fontweight="bold")
    axes[0].set_title("training cloud, fitted contours, the three probes",
                      fontsize=9)

    P.scatter_labelled(axes[1], d.X_val, y=d.y_val,
                       feature_names=d.feature_names)
    P.density_contours(axes[1], lambda X: log_independent_pdf(X, mu, var),
                       (48, 96), (2, 19))
    axes[1].set_title("validation set: do the contours enclose the anomalies?",
                      fontsize=9)
    P.save(fig, "stage2_engines_2d.png")

    notice("""
    Every contour is an axis-aligned ellipse. That is not a plotting choice,
    it is the independence assumption showing itself: p(x) is a product of
    per-feature densities, so the level sets can only be ellipses lined up
    with the axes. The training cloud is visibly tilted, so the model is
    already wrong about the shape of normality, it is just not wrong enough
    to matter yet on this dataset.

    Probes 2 and 3 are the tell. Both sit about 2 sd from the mean in each
    coordinate, so no single feature is alarming and the independent model
    scores them almost the same. But engine 3 runs warm while vibrating LESS
    than usual, which breaks the pattern the whole fleet obeys, and the
    correlation-aware distance says so: d = 4.0 against 2.0. Only the
    COMBINATION is strange, and a product of marginals cannot express a
    combination.

    Try it: change the off-diagonal 3.2 in data.engine_data to 0.0 and re-run.
    The contours and the cloud now agree, and probe 3 stops being interesting.
    Then push it to 3.9 (nearly the 4.0 that would make Sigma singular) and
    watch the mismatch become severe.
    """)


if __name__ == "__main__":
    main()
