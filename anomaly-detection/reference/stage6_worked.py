#!/usr/bin/env python3
"""Worked solution to the stage-6 capstone.  Run it from the project root:

    python3 reference/stage6_worked.py

It walks the four combinations of {raw, log} x {diagonal, full covariance} and
then adds one engineered feature, reporting recall per failure mode at every
step so you can see which fix buys which mode.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stages"))

from _common import banner, notice, section  # noqa: E402
from stage6_capstone_solver import header, report  # noqa: E402

from anomdet import (fit_independent, log_independent_pdf, fit_correlated,  # noqa: E402
                     log_correlated_pdf)
from anomdet import data  # noqa: E402


def main():
    banner("Stage 6: worked solution",
           "Four models, then one engineered feature. Watch the m1/m2/m3 "
           "columns rather than the aggregate F1.")

    d = data.solver_runs()
    skew = d.meta["skewed_columns"]

    def tf(X):
        """Log-transform the heavy-tailed columns.  Everything here is
        positive by construction, so plain log is safe; with data that can hit
        zero you would use log(x + c) for a small c."""
        Z = X.astype(float).copy()
        Z[:, skew] = np.log10(Z[:, skew])
        return Z

    section("Step 1: the four combinations")
    header()

    mu, var = fit_independent(d.X_train)
    report("raw + diagonal", d, lambda X: log_independent_pdf(X, mu, var))

    muf, Sf = fit_correlated(d.X_train)
    report("raw + full cov", d, lambda X: log_correlated_pdf(X, muf, Sf))

    muL, varL = fit_independent(tf(d.X_train))
    report("log + diagonal", d, lambda X: log_independent_pdf(tf(X), muL, varL))

    mufL, SfL = fit_correlated(tf(d.X_train))
    r_best = report("log + full cov", d,
                    lambda X: log_correlated_pdf(tf(X), mufL, SfL))

    section("Step 2: what the transform did to the correlations")
    raw_c = np.corrcoef(d.X_train, rowvar=False)[0, 1]
    log_c = np.corrcoef(tf(d.X_train), rowvar=False)[0, 1]
    print(f"  corr(cond_est, iters)            raw: {raw_c:+.3f}")
    print(f"  corr(log10 cond_est, iters)      log: {log_c:+.3f}")
    print("\n  The generating process is iters = a + b*log10(cond) + noise, so the")
    print("  dependence is only linear AFTER the transform. That is why 'raw +")
    print("  full cov' is worthless and 'log + full cov' finds mode 2: the")
    print("  transform is a precondition for the covariance model, not an")
    print("  alternative to it.")

    section("Step 3: one engineered feature, for the same job by hand")
    # Regress iters on log10(cond) using the TRAINING data only, then feed the
    # residual as an extra coordinate.  This hands the diagonal model exactly
    # the conditional information it cannot otherwise represent, the same
    # information the full covariance extracts, but as a single column.
    Ztr = tf(d.X_train)
    A = np.column_stack([np.ones(len(Ztr)), Ztr[:, 0]])
    coef, *_ = np.linalg.lstsq(A, Ztr[:, 1], rcond=None)
    print(f"  fitted on the training set:  iters ~ {coef[0]:.2f} "
          f"{coef[1]:+.2f} * log10(cond)")

    def tf_plus(X):
        Z = tf(X)
        resid = Z[:, 1] - (coef[0] + coef[1] * Z[:, 0])
        return np.column_stack([Z, resid])

    muP, varP = fit_independent(tf_plus(d.X_train))
    print()
    header()
    r_eng = report("log + resid + diagonal", d,
                   lambda X: log_independent_pdf(tf_plus(X), muP, varP))
    mufP, SfP = fit_correlated(tf_plus(d.X_train))
    r_all = report("log + resid + full cov", d,
                   lambda X: log_correlated_pdf(tf_plus(X), mufP, SfP))

    section("verdict against the stated target")
    target_ok = []
    for r in (r_best, r_eng, r_all):
        ok = r["f1"] >= 0.90 and min(r["per_mode"].values()) >= 0.85
        target_ok.append((r["name"], r["f1"], min(r["per_mode"].values()), ok))
    print(f"  {'model':<26}{'test F1':>9}{'worst mode recall':>20}   target met?")
    for name, f1, worst, ok in target_ok:
        print(f"  {name:<26}{f1:9.3f}{worst:20.3f}   {'YES' if ok else 'no'}")

    notice("""
    Two different routes reach the target. 'log + full covariance' lets the
    model discover the iters-versus-conditioning relationship from the data.
    'log + residual + diagonal' computes that relationship yourself and hands
    the model a single coordinate that already encodes it, and then the
    cheap independent model, the one from stage 2, is enough.

    The second route is usually the better engineering choice. It needs O(n)
    parameters instead of O(n^2), it keeps working when m is small, the new
    coordinate is interpretable to whoever gets paged ("this run took 2.1x
    the iterations its conditioning predicts"), and you can state its units.
    Reach for a richer model when you cannot name the structure; when you can
    name it, build the feature.
    """)


if __name__ == "__main__":
    main()
