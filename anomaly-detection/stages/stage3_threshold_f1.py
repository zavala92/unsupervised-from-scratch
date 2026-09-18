#!/usr/bin/env python3
"""Stage 3: choosing epsilon, and why accuracy is the wrong scoreboard.

Needs: fit_independent, independent_pdf, precision_recall_f1, tune_threshold
"""

import matplotlib.pyplot as plt
import numpy as np

from _common import banner, guard, notice, section

from anomdet import (fit_independent, independent_pdf, precision_recall_f1,
                     tune_threshold, tune_threshold_log, log_independent_pdf)
from anomdet import data, plotting as P


@guard
def main():
    banner("Stage 3: picking the threshold",
           "The model gives you p(x). Turning that into a decision needs one "
           "number, epsilon, and that number has to be earned on labelled "
           "data. This stage is about how to choose it and how to report it "
           "without fooling yourself.")

    d = data.engine_data()
    mu, var = fit_independent(d.X_train)
    p_val = independent_pdf(d.X_val, mu, var)
    p_test = independent_pdf(d.X_test, mu, var)
    y_val, y_test = d.y_val, d.y_test

    section("why not accuracy?")
    n_anom, n_tot = int(y_val.sum()), len(y_val)
    base = 1.0 - n_anom / n_tot
    print(f"  validation set: {n_tot} examples, {n_anom} of them anomalous "
          f"({100*n_anom/n_tot:.1f}%)")
    print(f"  a detector that flags NOTHING scores {100*base:.2f}% accuracy")
    print(f"  ... and has recall 0.000, precision 0.000, F1 0.000")
    print("  Accuracy cannot distinguish a working detector from a switched-off")
    print("  one. In real fraud or manufacturing settings anomalies are 0.1% or")
    print("  less, and the gap gets correspondingly more absurd.")

    section("select epsilon on the validation set")
    eps, f1_val = tune_threshold(y_val, p_val)
    print(f"  best epsilon = {eps:.6e}")
    print(f"  best F1 on validation = {f1_val:.4f}")
    print("  (that is the course's recipe: 1000 steps spaced uniformly in p.")
    print("   Hold on to the number, the log-spaced sweep below beats it.)")

    section("the trade-off curve, swept on a LOG grid this time")
    grid = np.exp(np.linspace(np.log(p_val.min()), np.log(p_val.max()), 400))
    prec, rec, f1s, accs = [], [], [], []
    for e in grid:
        pr, rc, f1 = precision_recall_f1(y_val, p_val < e)
        prec.append(pr); rec.append(rc); f1s.append(f1)
        accs.append(np.mean((p_val < e) == y_val.astype(bool)))
    prec, rec, f1s, accs = map(np.array, (prec, rec, f1s, accs))

    print(f"  {'epsilon':>12}  {'precision':>9}  {'recall':>7}  {'F1':>6}  {'accuracy':>8}")
    for e in [grid[5], grid[120], grid[np.argmax(f1s)], grid[300], grid[-5]]:
        pr, rc, f1 = precision_recall_f1(y_val, p_val < e)
        ac = np.mean((p_val < e) == y_val.astype(bool))
        star = "  <- best F1" if abs(e - grid[np.argmax(f1s)]) < 1e-30 else ""
        print(f"  {e:12.3e}  {pr:9.3f}  {rc:7.3f}  {f1:6.3f}  {ac:8.4f}{star}")

    section("now score it ONCE on the test set")
    pr, rc, f1 = precision_recall_f1(y_test, p_test < eps)
    print(f"  precision {pr:.3f}   recall {rc:.3f}   F1 {f1:.3f}")
    print(f"  (validation F1 was {f1_val:.3f}, the drop is the price of having")
    print("   tuned epsilon on the validation set. Report the test number.)")

    section("linear grid vs log grid")
    eps_lin, f1_lin = tune_threshold(y_val, p_val, n_steps=50)
    t_log, f1_log = tune_threshold_log(y_val, log_independent_pdf(d.X_val, mu, var),
                                         n_steps=50)
    print(f"  50 steps, uniform in p:      F1 = {f1_lin:.4f}")
    print(f"  50 steps, uniform in log p:  F1 = {f1_log:.4f}")
    print(f"\n  p on this set spans a factor of {p_val.max()/p_val.min():.2e}. A grid")
    print("  uniform in p puts essentially every node in the top decade, where")
    print("  nothing is being decided, and resolves the tail with a couple of")
    print("  points. That is why the log sweep in the table above found F1 =")
    print(f"  {f1s.max():.3f} while the linear search at 1000 steps settled for")
    print(f"  {f1_val:.3f}. The gap is already visible at n = 2; by stage 6, where")
    print("  log p spans hundreds of units, the linear grid is unusable.")

    section("figure")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].semilogx(grid, f1s, color="#d7191c", lw=1.8, label="F1")
    axes[0].semilogx(grid, prec, color="#2c7fb8", lw=1.2, ls="--", label="precision")
    axes[0].semilogx(grid, rec, color="#31a354", lw=1.2, ls=":", label="recall")
    axes[0].axvline(eps, color="k", lw=.9, alpha=.6)
    axes[0].set_xlabel("$\\epsilon$"); axes[0].set_ylabel("score")
    axes[0].set_title("the choice of $\\epsilon$ is a trade-off", fontsize=9)
    axes[0].legend(fontsize=8)

    axes[1].semilogx(grid, accs, color="#756bb1", lw=1.8)
    axes[1].axhline(base, color="k", ls="--", lw=1)
    axes[1].annotate("flag nothing", (grid[-1], base), fontsize=8,
                     ha="right", va="bottom")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("$\\epsilon$"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("accuracy: almost flat, almost useless", fontsize=9)

    P.scatter_labelled(axes[2], d.X_test, y=y_test, flagged=(p_test < eps),
                       feature_names=d.feature_names)
    P.density_contours(axes[2], lambda X: log_independent_pdf(X, mu, var),
                       (48, 96), (2, 19))
    axes[2].set_title(f"test set at the chosen $\\epsilon$ (F1 = {f1:.2f})",
                      fontsize=9)
    P.save(fig, "stage3_threshold_f1.png")

    notice("""
    The three-curve panel is the whole story of epsilon. Raising it buys
    recall and sells precision, monotonically, and F1 is just one particular
    opinion about the exchange rate. If a missed anomaly costs far more than a
    needless inspection, which is exactly the aircraft-engine case, then F1
    is the wrong objective and you should be maximising recall subject to a
    false-alarm budget you can staff.

    The accuracy panel is nearly flat at the base rate across four decades of
    epsilon. That is what an uninformative metric looks like.

    The two threshold searches disagree, and the log-spaced one wins. Spacing
    a search grid uniformly in a quantity that spans twelve orders of
    magnitude wastes almost all of it. This is a numerical-methods reflex
    rather than a machine-learning one, and it is the reason
    tune_threshold_log exists.

    Notice that the test F1 is lower than the validation F1. Nothing went
    wrong; epsilon was fitted on the validation set, so the validation score
    is optimistically biased. This is why the project keeps three splits:
    train (fit p), validation (choose epsilon), test (report). Consult the
    test set once.
    """)


if __name__ == "__main__":
    main()
