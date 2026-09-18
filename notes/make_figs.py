"""Generate every illustration for notes.tex as a vector PDF.

All numbers come from actual computation, with no schematic hand-drawing except
the confusion-matrix diagram, which is definitional.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "anomaly-detection"))
sys.path.insert(0, str(ROOT / "kmeans"))
import os
os.environ["ANOMDET_USE_REFERENCE"] = "1"

from kmeans import kmeans
from anomdet import (data, fit_independent, independent_pdf, log_independent_pdf,
                     precision_recall_f1, tune_threshold_log,
                     fit_correlated, log_correlated_pdf)

FIG = Path(__file__).resolve().parent / "figs"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": .7,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": .02,
})

BLUE, RED, ORANGE, GREEN, PURPLE = "#2c7fb8", "#d7191c", "#e6a000", "#31a354", "#756bb1"


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("  ", name)


def contours(ax, logp, xlim, ylim, decades=(1, 3, 6, 10, 16), color="0.4", ng=200):
    gx, gy = np.linspace(*xlim, ng), np.linspace(*ylim, ng)
    GX, GY = np.meshgrid(gx, gy)
    L = logp(np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    lv = sorted(L.max() - np.log(10.0) * np.array(decades, float))
    cs = ax.contour(GX, GY, L, levels=lv, colors=color, linewidths=.65)
    ax.clabel(cs, inline=True, fontsize=5.5,
              fmt={l: f"$10^{{-{d}}}$" for l, d in zip(lv, sorted(decades, reverse=True))})


# ====================================================== 1. K-MEANS, STEP BY STEP
def fig_kmeans():
    rng = np.random.default_rng(3)
    centres = np.array([[1.2, 1.4], [4.2, 1.0], [2.6, 4.3]])
    X = np.vstack([c + rng.standard_normal((45, 2)) * .62 for c in centres])

    mu = X[rng.choice(len(X), 3, replace=False)]        # init from examples
    cols = [BLUE, RED, GREEN]

    def assign(mu):
        return np.argmin(((X[:, None, :] - mu) ** 2).sum(2), axis=1)

    def distortion(idx, mu):
        return np.mean(((X - mu[idx]) ** 2).sum(1))

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.05), sharex=True, sharey=True)

    axes[0].scatter(*X.T, s=7, c="0.55", alpha=.8)
    axes[0].scatter(*mu.T, marker="X", s=90, c=cols, edgecolors="k", linewidths=.6)
    axes[0].set_title("(a) initialise $\\mu_k$\nat random examples")

    idx = assign(mu)
    for k in range(3):
        axes[1].scatter(*X[idx == k].T, s=7, c=cols[k], alpha=.8)
    axes[1].scatter(*mu.T, marker="X", s=90, c=cols, edgecolors="k", linewidths=.6)
    axes[1].set_title(f"(b) assign step\n$J = {distortion(idx, mu):.2f}$")

    mu_new = np.array([X[idx == k].mean(0) for k in range(3)])
    for k in range(3):
        axes[2].scatter(*X[idx == k].T, s=7, c=cols[k], alpha=.8)
        axes[2].annotate("", xy=mu_new[k], xytext=mu[k],
                         arrowprops=dict(arrowstyle="->", lw=1.1, color="k"))
    axes[2].scatter(*mu_new.T, marker="X", s=90, c=cols, edgecolors="k", linewidths=.6)
    axes[2].set_title(f"(c) move step\n$J = {distortion(idx, mu_new):.2f}$")

    mu_c = mu_new.copy()
    for _ in range(40):
        i = assign(mu_c)
        mu_c = np.array([X[i == k].mean(0) if (i == k).any() else mu_c[k]
                         for k in range(3)])
    i = assign(mu_c)
    for k in range(3):
        axes[3].scatter(*X[i == k].T, s=7, c=cols[k], alpha=.8)
    axes[3].scatter(*mu_c.T, marker="X", s=90, c=cols, edgecolors="k", linewidths=.6)
    axes[3].set_title(f"(d) converged\n$J = {distortion(i, mu_c):.2f}$")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    save(fig, "fig_kmeans.pdf")


# ================================================== 2. THE ANOMALY-DETECTION IDEA
def fig_concept():
    d = data.engine_data()
    mu, var = fit_independent(d.X_train)
    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    ax.scatter(*d.X_train.T, s=7, c=BLUE, alpha=.55,
               label="400 normal engines seen so far")
    contours(ax, lambda X: log_independent_pdf(X, mu, var), (49, 95), (2.5, 19))

    probes = np.array([[72.5, 10.8], [86.0, 4.2]])
    p = independent_pdf(probes, mu, var)
    ax.scatter(*probes[0], marker="*", s=170, c=GREEN, edgecolors="k",
               linewidths=.5, zorder=6)
    ax.scatter(*probes[1], marker="*", s=170, c=RED, edgecolors="k",
               linewidths=.5, zorder=6)
    ax.annotate(f"$p = {p[0]:.1e}$\nship it", xy=probes[0], xytext=(81.5, 15.2),
                fontsize=6.8, ha="left", zorder=7,
                bbox=dict(fc="white", ec="none", alpha=.85, pad=1.4),
                arrowprops=dict(arrowstyle="->", lw=.8, color="k",
                                shrinkA=2, shrinkB=5))
    ax.annotate(f"$p = {p[1]:.0e}$\ninspect it", xy=probes[1], xytext=(69.0, 3.4),
                fontsize=6.8, ha="left", zorder=7,
                bbox=dict(fc="white", ec="none", alpha=.85, pad=1.4),
                arrowprops=dict(arrowstyle="->", lw=.8, color="k",
                                shrinkA=2, shrinkB=5))
    ax.set_xlim(49, 95); ax.set_ylim(2.5, 19)
    ax.set_xlabel("$x_1$: heat [$^\\circ$C]")
    ax.set_ylabel("$x_2$: vibration [mm/s]")
    ax.legend(loc="upper left", fontsize=6.5, framealpha=.9)
    save(fig, "fig_concept.pdf")


# ============================================ 3. THE 1-D MODEL AND THE THRESHOLD
def fig_gaussian1d():
    d = data.engine_heat_1d()
    x = d.X_train[:, 0]
    mu, var = fit_independent(d.X_train)
    y = d.y_val.astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))

    axes[0].hist(x, bins=40, density=True, color=BLUE, alpha=.5,
                 edgecolor="white", linewidth=.3)
    g = np.linspace(x.min() - 6, x.max() + 6, 400)
    axes[0].plot(g, independent_pdf(g[:, None], mu, var), color=RED, lw=1.5)
    axes[0].axvline(mu[0], color=RED, ls=":", lw=.9)
    axes[0].set_xlabel("heat [$^\\circ$C]"); axes[0].set_ylabel("density")
    axes[0].set_title(f"fitted $\\mu={mu[0]:.1f}$, $\\sigma={np.sqrt(var[0]):.2f}$"
                      "  (true 70.0, 4.00)")

    axes[1].semilogy(g, independent_pdf(g[:, None], mu, var), color="0.35", lw=1.4)
    axes[1].scatter(d.X_val[~y, 0], independent_pdf(d.X_val[~y], mu, var), s=9,
                    c=BLUE, alpha=.65, label="normal")
    axes[1].scatter(d.X_val[y, 0], independent_pdf(d.X_val[y], mu, var), s=34,
                    facecolors="none", edgecolors=RED, linewidths=1.1,
                    label="anomaly")
    for eps, c, ls in [(1e-3, ORANGE, "--"), (1e-6, PURPLE, "-.")]:
        axes[1].axhline(eps, color=c, ls=ls, lw=1,
                        label=f"$\\varepsilon=10^{{{int(np.log10(eps))}}}$")
    axes[1].set_ylim(1e-13, 1); axes[1].set_xlabel("heat [$^\\circ$C]")
    axes[1].set_ylabel("$p(x)$")
    axes[1].set_title("the same model on a log axis:\nwhere the decision happens")
    axes[1].legend(fontsize=6, loc="lower center", ncol=2)
    save(fig, "fig_gaussian1d.pdf")


# ================================================ 4. CONFUSION MATRIX (schematic)
def fig_confusion():
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    cells = [((0, 1), "tp", GREEN, "caught it"),
             ((1, 1), "fp", ORANGE, "false alarm"),
             ((0, 0), "fn", RED, "missed it"),
             ((1, 0), "tn", "0.75", "never used")]
    for (cx, cy), lab, col, sub in cells:
        ax.add_patch(plt.Rectangle((cx, cy), 1, 1, facecolor=col, alpha=.3,
                                   edgecolor="k", lw=.8))
        ax.text(cx + .5, cy + .62, lab, ha="center", fontsize=12,
                style="italic", fontweight="bold")
        ax.text(cx + .5, cy + .27, sub, ha="center", fontsize=6.5)
    ax.text(.5, 2.12, "truly anomalous", ha="center", fontsize=7.5)
    ax.text(1.5, 2.12, "truly normal", ha="center", fontsize=7.5)
    ax.text(-.08, 1.5, "flagged", ha="right", va="center", fontsize=7.5)
    ax.text(-.08, .5, "not flagged", ha="right", va="center", fontsize=7.5)
    ax.text(1.0, -.42, "precision $=\\frac{tp}{tp+fp}$ (columns of the top row)",
            ha="center", fontsize=7)
    ax.text(1.0, -.78, "recall $=\\frac{tp}{tp+fn}$ (rows of the left column)",
            ha="center", fontsize=7)
    ax.set_xlim(-.95, 2.05); ax.set_ylim(-.95, 2.3); ax.axis("off")
    save(fig, "fig_confusion.pdf")


# ============================================== 5. THE PRECISION-RECALL TRADE-OFF
def fig_tradeoff():
    d = data.engine_data()
    mu, var = fit_independent(d.X_train)
    p = independent_pdf(d.X_val, mu, var)
    y = d.y_val
    grid = np.exp(np.linspace(np.log(p.min()), np.log(p.max()), 400))
    P, R, F, A = [], [], [], []
    for e in grid:
        pr, rc, f1 = precision_recall_f1(y, p < e)
        P.append(pr); R.append(rc); F.append(f1)
        A.append(np.mean((p < e) == y.astype(bool)))
    P, R, F, A = map(np.array, (P, R, F, A))
    best = grid[np.argmax(F)]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))
    axes[0].semilogx(grid, F, color=RED, lw=1.7, label="$F_1$")
    axes[0].semilogx(grid, P, color=BLUE, lw=1.1, ls="--", label="precision")
    axes[0].semilogx(grid, R, color=GREEN, lw=1.1, ls=":", label="recall")
    axes[0].axvline(best, color="k", lw=.8, alpha=.6)
    axes[0].annotate(f"best $F_1={F.max():.2f}$", (best, .06), fontsize=6.5,
                     ha="left")
    axes[0].set_xlabel("$\\varepsilon$"); axes[0].set_ylabel("score")
    axes[0].set_title("raising $\\varepsilon$ buys recall, sells precision")
    axes[0].legend(fontsize=6.5, loc="center left")

    base = 1 - y.mean()
    axes[1].semilogx(grid, A, color=PURPLE, lw=1.7)
    axes[1].axhline(base, color="k", ls="--", lw=.9)
    axes[1].annotate(f"flag nothing: {100*base:.1f}%", (grid[-1], base - .04),
                     ha="right", fontsize=6.5)
    axes[1].set_ylim(0, 1.03)
    axes[1].set_xlabel("$\\varepsilon$"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("accuracy: nearly flat, nearly useless")
    save(fig, "fig_tradeoff.pdf")


# ================================================== 6. INDEPENDENCE vs COVARIANCE
def fig_covariance():
    d = data.correlated_data()
    mu_d, var_d = fit_independent(d.X_train)
    mu_f, S = fit_correlated(d.X_train)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35), sharex=True, sharey=True)
    for ax, (name, lp) in zip(axes, [
            ("diagonal $\\Sigma$ (product of marginals)",
             lambda X: log_independent_pdf(X, mu_d, var_d)),
            ("full $\\Sigma$", lambda X: log_correlated_pdf(X, mu_f, S))]):
        t, _ = tune_threshold_log(d.y_val, lp(d.X_val), n_steps=4000)
        pred = lp(d.X_test) < t
        pr, rc, f1 = precision_recall_f1(d.y_test, pred)
        yb = d.y_test.astype(bool)
        ax.scatter(*d.X_test[~yb].T, s=7, c=BLUE, alpha=.55, label="normal")
        ax.scatter(*d.X_test[yb].T, s=52, facecolors="none", edgecolors=RED,
                   linewidths=1.3, label="true anomaly")
        ax.scatter(*d.X_test[pred].T, s=34, marker="x", c=ORANGE, linewidths=1.2,
                   label="flagged")
        contours(ax, lp, (-3.2, 3.2), (-3.2, 3.2), decades=(1, 2, 4, 7, 11))
        ax.set_title(f"{name}\ntest $F_1 = {f1:.3f}$  (P {pr:.2f} / R {rc:.2f})")
        ax.set_xlabel("$x_1$"); ax.set_aspect("equal")
    axes[0].set_ylabel("$x_2$")
    axes[0].legend(fontsize=6.5, loc="upper left")
    save(fig, "fig_covariance.pdf")


# ================================================== 7. FEATURE TRANSFORM + UNDERFLOW
def fig_numerics():
    d = data.skewed_server_data()
    lat = d.X_train[:, 0]
    mu_r, var_r = fit_independent(d.X_train)
    llat = np.log(lat)
    mu_l, var_l = fit_independent(np.column_stack([llat, d.X_train[:, 1]]))

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.25))
    fig.subplots_adjust(wspace=0.40)

    axes[0].hist(lat, bins=45, density=True, color=BLUE, alpha=.5,
                 edgecolor="white", linewidth=.3)
    axes[0].set_xlim(-330, 1000)
    g = np.linspace(-330, 1000, 400)
    axes[0].plot(g, np.exp(-(g - mu_r[0])**2 / (2*var_r[0])) /
                 np.sqrt(2*np.pi*var_r[0]), color=RED, lw=1.5)
    axes[0].axvspan(-330, 0, color=RED, alpha=.11, lw=0)
    axes[0].axvline(0, color="k", lw=.9)
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.30)
    axes[0].annotate("11% of the fitted mass\nlands in the shaded region",
                     xy=(0.035, 0.965), xycoords="axes fraction",
                     fontsize=5.9, va="top", ha="left", color=RED)
    axes[0].set_xlabel("latency [ms]"); axes[0].set_ylabel("density")
    axes[0].set_title("(a) skewed feature, fitted raw")

    axes[1].hist(llat, bins=45, density=True, color=BLUE, alpha=.5,
                 edgecolor="white", linewidth=.3)
    lo, hi = axes[1].get_xlim()
    g = np.linspace(lo, hi, 400)
    axes[1].plot(g, np.exp(-(g - mu_l[0])**2 / (2*var_l[0])) /
                 np.sqrt(2*np.pi*var_l[0]), color=RED, lw=1.5)
    axes[1].set_xlabel("$\\log$(latency)")
    axes[1].set_title("(b) after $x \\mapsto \\log x$")

    ns = np.array([2, 50, 100, 200, 300, 400, 500, 520, 560, 600, 700, 800])
    pm = []
    for n in ns:
        X = data.high_dim_normal(n=n)
        pm.append(independent_pdf(X, np.zeros(n), np.ones(n)).max())
    pm = np.array(pm); ok = pm > 0
    axes[2].semilogy(ns[ok], pm[ok], "o-", color=BLUE, ms=3.2, lw=1.1,
                     label="$\\max p(x)$")
    axes[2].semilogy(ns[~ok], np.full((~ok).sum(), 2e-323), "x", color=RED,
                     ms=6, label="$p(x)=0$ exactly")
    axes[2].axhline(5e-324, color="k", ls="--", lw=.9)
    axes[2].set_ylim(1e-323, 1e5)
    axes[2].set_yticks([1e0, 1e-80, 1e-160, 1e-240, 1e-320])
    axes[2].tick_params(axis="y", labelsize=6.4, pad=1.5)
    axes[2].annotate("smallest positive double", xy=(0.03, 0.045),
                     xycoords="axes fraction", fontsize=5.6, va="bottom")
    axes[2].set_xlabel("number of features $n$")
    axes[2].set_ylabel("$\\max p(x)$")
    axes[2].set_title("(c) the underflow cliff")
    axes[2].legend(fontsize=6, loc="upper right")
    save(fig, "fig_numerics.pdf")


def _part2():
    fig_concept(); fig_gaussian1d(); fig_confusion()
    fig_tradeoff(); fig_covariance(); fig_numerics()


# =============================================================================
#                     CLUSTERING FIGURES (added for Part I)
# =============================================================================

def _kmeans(X, K, rng, iters=100, tol=0):
    """Thin adapter onto kmeans.kmeans so the figures and the library cannot
    drift apart."""
    return kmeans(X, K, rng, max_iters=iters, tol=tol)


# ------------------------------------------- local optima and multiple restarts
def fig_localoptima():
    rng0 = np.random.default_rng(11)
    centres = np.array([[0, 0], [3.3, .3], [1.7, 3.0], [5.2, 3.2]])
    X = np.vstack([c + rng0.standard_normal((40, 2)) * .52 for c in centres])

    runs = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        mu, idx, hist = _kmeans(X, 4, rng)
        runs.append((hist[-1], mu, idx, hist, seed))
    runs.sort(key=lambda r: r[0])
    best, worst = runs[0], runs[-1]

    fig = plt.figure(figsize=(7.4, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=.28)

    ax = fig.add_subplot(gs[0])
    n_stuck = sum(1 for r in runs if r[0] > runs[0][0] * 1.05)
    first_bad = next(i for i, r in enumerate(runs) if r[0] > runs[0][0] * 1.05)
    shown = [runs[0], runs[first_bad], runs[-1]]
    for (Jf, _, _, hist, sd), c, lab in zip(shown, [GREEN, ORANGE, RED],
                                            [f"global opt. ({len(runs)-n_stuck}/40 runs)",
                                             "a bad local optimum",
                                             "worst of 40"]):
        ax.plot(np.arange(len(hist)) / 2, hist, color=c, lw=1.3,
                label=f"{lab}: $J={Jf:.2f}$")
    ax.set_xlabel("iteration"); ax.set_ylabel("distortion $J$")
    ax.set_title("$J$ never increases, but\nit stops at different values")
    print(f"     [local optima: {n_stuck} of 40 restarts got stuck]")
    ax.legend(fontsize=6.2)
    ax.set_xlim(0, 7)

    cols = [BLUE, RED, GREEN, PURPLE]
    for gi, (tag, run) in enumerate([("worst of 40", worst), ("best of 40", best)]):
        ax = fig.add_subplot(gs[gi + 1])
        Jf, mu, idx, _, _ = run
        for k in range(4):
            ax.scatter(*X[idx == k].T, s=6, c=cols[k], alpha=.85)
        ax.scatter(*mu.T, marker="X", s=75, c=cols, edgecolors="k", linewidths=.55)
        ax.set_title(f"{tag}:  $J = {Jf:.2f}$")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    save(fig, "fig_localoptima.pdf")


# --------------------------------------------------------- choosing K: the elbow
def fig_elbow():
    rng = np.random.default_rng(5)
    centres = np.array([[0, 0], [4.0, .4], [2.0, 3.6]])
    X_clust = np.vstack([c + rng.standard_normal((60, 2)) * .45 for c in centres])
    X_blob = rng.standard_normal((180, 2)) * np.array([1.9, 1.0])

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.25))
    axes[0].scatter(*X_clust.T, s=6, c="0.4", alpha=.8)
    axes[1].scatter(*X_blob.T, s=6, c="0.4", alpha=.8)
    for ax, lab in zip(axes[:2], ["(a) three real clusters",
                                  "(b) one elongated blob"]):
        # identical limits so the two equal-aspect boxes have identical shape,
        # which keeps the panel labels on the same line
        ax.set_xlim(-5.6, 5.9); ax.set_ylim(-3.3, 5.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.annotate(lab, xy=(0.5, 1.04), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=8.5)

    Ks = np.arange(1, 9)
    for X, c, lab, m in [(X_clust, GREEN, "(a) three clusters", "o"),
                         (X_blob, RED, "(b) one blob", "s")]:
        J = []
        for K in Ks:
            best = min(_kmeans(X, K, np.random.default_rng(s))[2][-1]
                       for s in range(25))
            J.append(best)
        axes[2].plot(Ks, J, m + "-", color=c, ms=3.4, lw=1.2, label=lab)
    axes[2].axvline(3, color=GREEN, ls=":", lw=1)
    axes[2].annotate("elbow", (3.1, axes[2].get_ylim()[1] * .62), fontsize=6.5,
                     color=GREEN)
    axes[2].set_xlabel("$K$"); axes[2].set_ylabel("best $J$ of 25 restarts")
    axes[2].set_title("(c) $J$ versus $K$")
    axes[2].legend(fontsize=6.2)
    save(fig, "fig_elbow.pdf")


# ------------------------------------------ K-means for colour quantisation
def fig_compression():
    import matplotlib.cbook as cbook
    img = plt.imread(cbook.get_sample_data("grace_hopper.jpg")).astype(float) / 255.
    h, w, _ = img.shape
    px = img.reshape(-1, 3)
    n_unique = len(np.unique((px * 255).astype(np.uint8), axis=0))

    rng = np.random.default_rng(0)
    sub = px[rng.choice(len(px), 8000, replace=False)]        # fit on a subsample
    K = 16
    mu, _, _ = _kmeans(sub, K, np.random.default_rng(2), iters=40)
    idx = np.argmin(((px[:, None, :] - mu) ** 2).sum(2), axis=1)
    recon = mu[idx].reshape(h, w, 3)

    orig_bits = h * w * 24
    comp_bits = h * w * int(np.ceil(np.log2(K))) + K * 24

    fig = plt.figure(figsize=(7.4, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=.18)

    ax = fig.add_subplot(gs[0]); ax.imshow(img); ax.axis("off")
    ax.set_title(f"original\n{n_unique:,} distinct colours")
    ax = fig.add_subplot(gs[1]); ax.imshow(np.clip(recon, 0, 1)); ax.axis("off")
    ax.set_title(f"$K={K}$ colours\n{orig_bits/comp_bits:.1f}$\\times$ smaller")

    ax = fig.add_subplot(gs[2], projection="3d")
    s = px[rng.choice(len(px), 4000, replace=False)]
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], c=s, s=3.2, alpha=.55, linewidths=0)
    ax.scatter(mu[:, 0], mu[:, 1], mu[:, 2], c=mu, s=95, marker="o",
               edgecolors="k", linewidths=.7, depthshade=False)
    ax.set_xlabel("R", labelpad=-7); ax.set_ylabel("G", labelpad=-7)
    # set_zlabel does not render reliably at this view angle; place it by hand.
    ax.text2D(0.985, 0.47, "B", transform=ax.transAxes, fontsize=8.5,
              ha="center", va="center")
    ax.tick_params(labelsize=5, pad=-3)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_zticks([0, 1])
    ax.set_title("pixels in RGB space,\nwith the 16 centroids", pad=-2)
    ax.view_init(elev=18, azim=-58)
    save(fig, "fig_compression.pdf")
    return n_unique, orig_bits / comp_bits


if __name__ == "__main__":
    print("writing figures:")
    fig_kmeans(); fig_localoptima(); fig_elbow(); fig_compression()   # Part I
    _part2()                                                          # Part II
