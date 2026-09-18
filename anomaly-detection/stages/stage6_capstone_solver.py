#!/usr/bin/env python3
"""Stage 6: the capstone. Six features of solver telemetry, three planted
failure modes, and no hand-holding.

This script gives you the data, an honest scoreboard, and a baseline to beat.
The model is YOUR job: everything between the two BUILD YOUR DETECTOR markers.

    python3 stages/stage6_capstone_solver.py

Target: overall test F1 >= 0.90 with recall >= 0.85 on EACH of the three
failure modes. The aggregate alone is not enough, mode 2 is only 8 of 424
examples, so you can score 0.74 overall while being completely blind to it.

The three modes each need a different lesson from stages 1-5. Which one is
which is written down in anomdet/data.SOLVER_FAILURE_MODES, read it after
you have tried, not before.
"""

import matplotlib.pyplot as plt
import numpy as np

from _common import banner, guard, notice, section

from anomdet import (fit_independent, log_independent_pdf, fit_correlated,
                     log_correlated_pdf, tune_threshold_log,
                     precision_recall_f1)
from anomdet import data, plotting as P


def report(name, d, logp_fn, n_steps=4000):
    """Fit-free evaluation harness: tune the cut on validation, score on test,
    and break recall down by failure mode."""
    t, f1_val = tune_threshold_log(d.y_val, logp_fn(d.X_val), n_steps=n_steps)
    pred = logp_fn(d.X_test) < t
    pr, rc, f1 = precision_recall_f1(d.y_test, pred)
    modes = d.meta["modes_test"]
    per_mode = {m: float(pred[modes == m].mean()) for m in (1, 2, 3)}
    fp = int(np.sum(pred & ~d.y_test.astype(bool)))
    print(f"  {name:<28}{f1_val:8.3f}{pr:9.3f}{rc:8.3f}{f1:8.3f}   "
          + " ".join(f"{per_mode[m]:5.2f}" for m in (1, 2, 3))
          + f"   {fp:4d}")
    return dict(name=name, threshold=t, f1_val=f1_val, precision=pr, recall=rc,
                f1=f1, per_mode=per_mode, pred=pred, logp_fn=logp_fn,
                false_positives=fp)


def header():
    print(f"  {'model':<28}{'val F1':>8}{'prec':>9}{'rec':>8}{'F1':>8}   "
          f"{'m1':>5} {'m2':>5} {'m3':>5}   {'FP':>4}")
    print("  " + "-" * 76)


@guard
def main():
    banner("Stage 6: capstone: is this solver run healthy?",
           "Telemetry from 900 healthy runs of an iterative linear solver. "
           "Six features, all of them physically meaningful, with latent "
           "problem size and conditioning inducing real correlations. Three "
           "failure modes are planted in the labelled splits.")

    d = data.solver_runs()
    print(f"\n  train {d.X_train.shape}   val {d.X_val.shape}   test {d.X_test.shape}"
          f"   production {d.meta['X_production'].shape} (unlabelled)")
    print(f"  anomalies: {int(d.y_val.sum())} in val, {int(d.y_test.sum())} in test,"
          f" evenly split across 3 modes")

    section("look at your features before you model them")
    print(f"  {'feature':<16}{'min':>12}{'median':>12}{'max':>12}"
          f"{'max/med':>10}{'skew':>8}")
    for j, name in enumerate(d.feature_names):
        c = d.X_train[:, j]
        sk = np.mean(((c - c.mean()) / c.std()) ** 3)
        print(f"  {name:<16}{c.min():12.3e}{np.median(c):12.3e}{c.max():12.3e}"
              f"{c.max()/np.median(c):10.1f}{sk:8.2f}")
    print("\n  Two of these span five orders of magnitude and have skewness > 3.")
    print("  Stage 4 says what to do about that.")

    print(f"\n  correlation matrix of the RAW features:")
    C = np.corrcoef(d.X_train, rowvar=False)
    print("      " + "".join(f"{n[:7]:>9}" for n in d.feature_names))
    for i, n in enumerate(d.feature_names):
        print(f"  {n[:7]:>7} " + "".join(f"{C[i,j]:9.2f}" for j in range(len(C))))
    print("\n  Now compute the same thing after log-transforming the skewed")
    print("  columns and compare. Some correlations are only LINEAR in log")
    print("  space, and a full-covariance model can only use linear ones.")

    section("baselines to beat")
    header()
    mu_d, var_d = fit_independent(d.X_train)
    base1 = report("raw + diagonal", d, lambda X: log_independent_pdf(X, mu_d, var_d))
    mu_f, S_f = fit_correlated(d.X_train)
    base2 = report("raw + full covariance", d,
                   lambda X: log_correlated_pdf(X, mu_f, S_f))

    lp = base1["logp_fn"](d.X_test)
    print(f"\n  Look at the scale the raw model works on: log p runs down to"
          f" {lp.min():.1e}.")
    print("  res_final has variance ~1e-21, so an ordinary deviation in that")
    print("  column produces a z^2 of order 1e12 and swamps every other feature.")
    print("  The raw model is not just mis-shaped, it is mis-SCALED: effectively")
    print("  it is a one-feature detector. Look at the left panel of the figure")
    print("  and you can see it, normal runs, mode 2 and mode 3 are all stacked")
    print("  in a single bin, indistinguishable.")

    # ==================================================================
    # ------------------------- BUILD YOUR DETECTOR -------------------------
    #
    # Everything you need is already imported. A reasonable plan:
    #
    #   1. Write a transform  tf(X) -> X'  that log-transforms the skewed
    #      columns.  d.meta["skewed_columns"] tells you which ones, though you
    #      should confirm it from the table above rather than trusting it.
    #   2. Fit on tf(d.X_train), diagonal first, then full covariance.
    #   3. Pass  lambda X: <your log density>(tf(X))  to report().
    #   4. Read the per-mode recall columns m1/m2/m3.  A mode stuck at 0.00 is
    #      telling you which lesson is still missing, not that the mode is
    #      impossible.
    #   5. If a mode stays stubborn, engineer a feature for it.  Stage 5's
    #      last paragraph is the hint: a ratio or a residual-from-regression
    #      turns a correlation the product model cannot see into a single
    #      coordinate it can.
    #
    # Delete the `pass` and write your code here.
    # ------------------------------------------------------------------

    pass

    # ----------------------- END BUILD YOUR DETECTOR -----------------------
    # ==================================================================

    section("scoring the production batch")
    Xp = d.meta["X_production"]
    best = max([base1, base2], key=lambda r: r["f1"])
    flagged = best["logp_fn"](Xp) < best["threshold"]
    print(f"  using the best model available in this script: {best['name']}")
    print(f"  flagged {flagged.sum()} of {len(Xp)} production runs "
          f"({100*flagged.mean():.1f}%) for inspection")
    print(f"  at precision {best['precision']:.2f} on the test set, roughly")
    print(f"  {best['precision']*flagged.sum():.0f} of those are expected to be real.")
    print("\n  Once your own detector beats the baseline, point this section at")
    print("  it and see how the inspection load changes. That number, how")
    print("  many runs a human has to look at per day, is what an operations")
    print("  team will actually ask you about.")

    section("figure")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    logp = best["logp_fn"](d.X_test)
    modes = d.meta["modes_test"]

    # log p under a badly scaled model can run to -1e12, which would squash
    # every interesting group against the axis.  Clip to a window around the
    # normal class and say so: the leftmost bin is a pile-up, not a count.
    normal_lp = logp[modes == 0]
    lo = np.percentile(normal_lp, 0.5) - 3.0 * (np.median(normal_lp) - np.percentile(normal_lp, 5))
    hi = logp.max()
    lo = min(lo, best["threshold"] - 0.05 * abs(hi - best["threshold"]) - 1e-9)
    bins = np.linspace(lo, hi, 55)
    clipped = np.clip(logp, lo, hi)
    n_below = int(np.sum(logp < lo))

    axes[0].hist(clipped[modes == 0], bins=bins, color=P.NORMAL_C, alpha=.65,
                 label=f"normal ({(modes==0).sum()})")
    for m, c in zip((1, 2, 3), ("#d7191c", "#fdae61", "#756bb1")):
        axes[0].hist(clipped[modes == m], bins=bins, color=c, alpha=.85,
                     label=f"mode {m} ({(modes==m).sum()})")
    axes[0].axvline(best["threshold"], color="k", ls="--", lw=1.3)
    axes[0].annotate("$\\epsilon$", (best["threshold"], axes[0].get_ylim()[1] * .9),
                     fontsize=10, ha="right")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("log p(x)"
                       + (f"   (clipped; {n_below} point(s) fall further left)"
                          if n_below else ""))
    axes[0].set_ylabel("count")
    axes[0].set_title(f"{best['name']}: where each mode lands", fontsize=9)
    axes[0].legend(fontsize=7)

    w = .35
    xs = np.arange(3)
    axes[1].bar(xs - w / 2, [base1["per_mode"][m] for m in (1, 2, 3)], w,
                label=base1["name"], color=P.NORMAL_C)
    axes[1].bar(xs + w / 2, [base2["per_mode"][m] for m in (1, 2, 3)], w,
                label=base2["name"], color=P.ANOM_C)
    axes[1].axhline(.85, color="k", ls=":", lw=1)
    axes[1].annotate("target", (2.4, .86), fontsize=8)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"mode {m}\n{data.SOLVER_FAILURE_MODES[m][0]}"
                             for m in (1, 2, 3)], fontsize=7)
    axes[1].set_ylabel("recall")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("recall per failure mode", fontsize=9)
    axes[1].legend(fontsize=8)
    P.save(fig, "stage6_capstone_solver.png")

    notice("""
    The baselines are deliberately bad, and bad in an instructive way: raw +
    diagonal and raw + full covariance score IDENTICALLY. Adding off-diagonal
    covariance to the raw features buys literally nothing, because the real
    dependence in this data is iters against LOG cond_est, and that is not a
    linear relationship in the raw coordinates. Covariance can only capture
    linear structure, so the transform is what makes the covariance useful.
    The two fixes are not independent improvements you can pick between --
    one of them is a precondition for the other.

    That is the most transferable thing in this project. 'Which model should
    I use' is almost always a less productive question than 'what coordinates
    am I looking at'.

    When you are done, read anomdet/data.SOLVER_FAILURE_MODES and check that
    your per-mode recalls line up with which lesson each mode was built to
    require. If mode 2 is still at zero, you have not yet given the model a
    way to see a conditional relationship.
    """)


if __name__ == "__main__":
    main()
