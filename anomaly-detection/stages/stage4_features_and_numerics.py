#!/usr/bin/env python3
"""Stage 4: two failures that have nothing to do with the model: the shape
of your features, and the arithmetic you evaluate them in.

Needs: fit_independent, independent_pdf, log_independent_pdf,
       precision_recall_f1, tune_threshold_log
"""

import matplotlib.pyplot as plt
import numpy as np

from _common import banner, guard, notice, section

from anomdet import (fit_independent, independent_pdf, log_independent_pdf,
                     precision_recall_f1, tune_threshold_log)
from anomdet import data, plotting as P


def fit_and_score(X_train, X_val, y_val, X_test, y_test):
    mu, var = fit_independent(X_train)
    t, f1_val = tune_threshold_log(y_val, log_independent_pdf(X_val, mu, var),
                                     n_steps=3000)
    pred = log_independent_pdf(X_test, mu, var) < t
    return (f1_val,) + precision_recall_f1(y_test, pred) + (mu, var, t, pred)


@guard
def main():
    banner("Stage 4: feature shape, and the arithmetic underneath",
           "Part A: a feature with a heavy tail wrecks a detector even when "
           "it carries no anomaly signal at all. Part B: the product form of "
           "p(x) underflows to exactly zero in double precision, and the "
           "detector dies silently.")

    # ------------------------------------------------------------- PART A
    section("Part A: a log-normal nuisance feature")
    d = data.skewed_server_data()
    print("  latency    is log-normal and carries NO anomaly signal")
    print("  throughput is Gaussian and is where the signal lives (4.5 sd low)")
    print(f"  {d.X_train.shape[0]} normal training servers, "
          f"{int(d.y_test.sum())} anomalies in the test set")

    lat = d.X_train[:, 0]
    print(f"\n  latency: median {np.median(lat):7.1f} ms, mean {lat.mean():7.1f} ms,"
          f" max {lat.max():8.1f} ms")
    print(f"  mean/median = {lat.mean()/np.median(lat):.2f}, a Gaussian would give 1.00")
    print(f"  skewness    = {np.mean(((lat-lat.mean())/lat.std())**3):.2f}"
          "; a Gaussian would give 0.00")

    raw = fit_and_score(d.X_train, d.X_val, d.y_val, d.X_test, d.y_test)
    mu_r, var_r = raw[4], raw[5]
    sd_r = np.sqrt(var_r[0])
    print(f"\n  Fitting a Gaussian straight onto latency gives mu = {mu_r[0]:.1f},"
          f" sd = {sd_r:.1f}.")
    from math import erf, sqrt
    frac_neg = 0.5 * (1.0 + erf((0.0 - mu_r[0]) / sd_r / sqrt(2.0)))
    print(f"  That model places {100*frac_neg:.0f}% of its probability mass on NEGATIVE")
    print("  latency. Not a rounding detail: an eighth of the model's belief is")
    print("  spent on a physically impossible region, and it is stolen from the")
    print("  region where the data actually are.")
    tail = lat[lat > mu_r[0] + 3 * sd_r]
    print(f"  Meanwhile {len(tail)} of {len(lat)} perfectly normal training servers"
          f" sit beyond 3 sd,")
    print("  where the model expects 0.1%. Those are the false positives to come.")

    def tlog(X):
        Z = X.copy()
        Z[:, 0] = np.log(Z[:, 0])
        return Z

    log = fit_and_score(tlog(d.X_train), tlog(d.X_val), d.y_val,
                        tlog(d.X_test), d.y_test)

    print(f"\n  {'features':<22}{'val F1':>8}{'precision':>11}{'recall':>8}{'test F1':>9}")
    for name, r in (("latency as given", raw), ("log(latency)", log)):
        print(f"  {name:<22}{r[0]:8.3f}{r[1]:11.3f}{r[2]:8.3f}{r[3]:9.3f}")

    print(f"\n  Recall is unchanged ({raw[2]:.3f} -> {log[2]:.3f}); precision goes")
    print(f"  {raw[1]:.3f} -> {log[1]:.3f}. Bad feature shape does not make you miss")
    print("  anomalies, it makes you drown in false alarms, and a detector")
    print("  nobody trusts gets switched off, which is the same as having none.")

    lat_log = np.log(lat)
    print(f"\n  after the transform: skewness = "
          f"{np.mean(((lat_log-lat_log.mean())/lat_log.std())**3):+.2f}")
    print("  Rule of thumb: plot a histogram of every feature before you model")
    print("  it. Try log(x), log(x + c), sqrt(x), x**0.3 and keep what looks")
    print("  symmetric. This costs minutes and is usually worth more than any")
    print("  amount of cleverness in the model.")

    # ------------------------------------------------------------- PART B
    section("Part B: p(x) underflows, log p does not")
    print(f"  {'n':>6}{'max p(x)':>14}{'all exactly 0?':>16}{'mean log p':>13}")
    for n in (2, 50, 200, 500, 520, 600, 800):
        X = data.high_dim_normal(n=n)
        mu, var = np.zeros(n), np.ones(n)
        p = independent_pdf(X, mu, var)
        lp = log_independent_pdf(X, mu, var)
        print(f"  {n:6d}{p.max():14.3e}{str(bool(np.all(p == 0))):>16}{lp.mean():13.1f}")

    print("\n  The smallest positive double is about 4.9e-324. A product of n")
    print("  standard-normal densities is roughly exp(-1.42 n), so it reaches")
    print("  that floor at n around 520, and there is no warning: p becomes")
    print("  0.0, every point ties for most anomalous, and `p < epsilon` flags")
    print("  the entire dataset. Sums of logs have no such cliff; the mean log p")
    print("  at n = 800 is about -1134, which is a perfectly ordinary float.")

    # ------------------------------------------------------------- figure
    section("figure")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    P.hist_with_normal_fit(axes[0], lat, mu_r[0], var_r[0],
                           title="latency as given: the fit is not credible",
                           xlim=(-320, 1000))
    axes[0].set_xlabel("latency [ms]")
    axes[0].axvline(0, color="k", lw=1.1)
    axes[0].axvspan(-320, 0, color="#d7191c", alpha=.10, lw=0)
    axes[0].annotate("the fitted Gaussian puts\n11% of its mass on\nNEGATIVE latency",
                     (-300, axes[0].get_ylim()[1] * .62), fontsize=7, ha="left",
                     va="top")

    mu_l, var_l = log[4], log[5]
    P.hist_with_normal_fit(axes[1], lat_log, mu_l[0], var_l[0],
                           title="log(latency): the fit is honest")
    axes[1].set_xlabel("log(latency)")

    ns = np.array([2, 50, 100, 200, 300, 400, 500, 520, 560, 600, 700, 800])
    pmax, lpmean = [], []
    for n in ns:
        X = data.high_dim_normal(n=n)
        mu, var = np.zeros(n), np.ones(n)
        pmax.append(independent_pdf(X, mu, var).max())
        lpmean.append(log_independent_pdf(X, mu, var).mean())
    pmax = np.array(pmax)
    ok = pmax > 0
    axes[2].semilogy(ns[ok], pmax[ok], "o-", color=P.NORMAL_C, label="max p(x)")
    axes[2].semilogy(ns[~ok], np.full((~ok).sum(), 1e-323), "x",
                     color=P.ANOM_C, ms=9, label="p(x) = 0 exactly")
    axes[2].axhline(5e-324, color="k", ls="--", lw=1)
    axes[2].annotate("smallest positive double", (ns[0], 5e-324), fontsize=7,
                     va="bottom")
    axes[2].set_xlabel("number of features $n$")
    axes[2].set_ylabel("max $p(x)$ over the sample")
    axes[2].set_title("the underflow cliff", fontsize=9)
    axes[2].legend(fontsize=8)
    P.save(fig, "stage4_features_and_numerics.png")

    notice("""
    Part A is the lesson people skip. The anomaly signal was never in latency
    The anomalous servers had exactly the same latency distribution as the
    healthy ones. Including that feature with the wrong shape still cost
    precision 1.00 -> 0.61, because the inflated variance made honest
    slow-but-fine servers look like 4-sigma events and they outranked the
    genuinely starved ones. A nuisance feature is not harmless if you model
    it badly.

    Part B is a bug you will not notice from the accuracy number. At n = 600
    every p(x) is 0.0, so `p < epsilon` is true for everything, so the
    detector flags 100% of traffic: recall 1.0, precision at the base rate.
    That looks like a badly tuned threshold, not like arithmetic. Work in
    log p and the failure mode simply does not exist.

    Try it: in part A, transform with sqrt(latency) or latency**0.3 instead of
    log. Which is the most symmetric, and does the ranking by skewness agree
    with the ranking by test F1?
    """)


if __name__ == "__main__":
    main()
