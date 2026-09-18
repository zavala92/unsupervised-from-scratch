"""Plot helpers.  Nothing here is required by the exercises, it exists so the
stages can show you what your detector is doing.  Every figure lands in
figures/ as a PNG.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGDIR = Path(__file__).resolve().parent.parent / "figures"
FIGDIR.mkdir(exist_ok=True)

NORMAL_C, ANOM_C, FLAG_C = "#2c7fb8", "#d7191c", "#fdae61"


def save(fig, name):
    path = FIGDIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> figures/{name}")
    return path


def scatter_labelled(ax, X, y=None, flagged=None, feature_names=None):
    """Blue = normal, red ring = true anomaly, orange x = flagged by model."""
    if y is None:
        ax.scatter(X[:, 0], X[:, 1], s=14, c=NORMAL_C, alpha=.6, label="data")
    else:
        y = np.asarray(y).astype(bool)
        ax.scatter(X[~y, 0], X[~y, 1], s=14, c=NORMAL_C, alpha=.6, label="normal")
        ax.scatter(X[y, 0], X[y, 1], s=70, facecolors="none", edgecolors=ANOM_C,
                   linewidths=1.8, label="true anomaly")
    if flagged is not None:
        flagged = np.asarray(flagged).astype(bool)
        ax.scatter(X[flagged, 0], X[flagged, 1], s=46, marker="x", c=FLAG_C,
                   linewidths=1.6, label="flagged")
    if feature_names:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    ax.legend(fontsize=8, loc="best")


def density_contours(ax, logp_fn, xlim, ylim, ngrid=220, decades=(1, 3, 6, 10, 16),
                     color="0.35"):
    """Contours of log p, drawn at fixed numbers of DECADES below the peak.

    Plotting log p rather than p is what makes the far tail visible at all:
    the level p = p_max * 1e-16 is where the interesting decisions happen and
    it is utterly invisible on a linear density scale.
    """
    gx = np.linspace(*xlim, ngrid)
    gy = np.linspace(*ylim, ngrid)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    L = logp_fn(pts).reshape(GX.shape)
    peak = L.max()
    levels = sorted(peak - np.log(10.0) * np.array(decades, dtype=float))
    cs = ax.contour(GX, GY, L, levels=levels, colors=color, linewidths=.8)
    ax.clabel(cs, inline=True, fontsize=6,
              fmt={lv: f"$10^{{-{d}}}$" for lv, d in zip(levels, sorted(decades, reverse=True))})
    return cs


def hist_with_normal_fit(ax, x, mu, var, bins=45, title=None, logx=False,
                         xlim=None):
    ax.hist(x, bins=bins, density=True, color=NORMAL_C, alpha=.55,
            edgecolor="white", linewidth=.4)
    if xlim is not None:
        ax.set_xlim(*xlim)
    lo, hi = ax.get_xlim()
    g = np.linspace(lo, hi, 400)
    pdf = np.exp(-(g - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
    ax.plot(g, pdf, color=ANOM_C, lw=1.8, label="fitted Gaussian")
    ax.axvline(mu, color=ANOM_C, ls=":", lw=1)
    if logx:
        ax.set_xscale("log")
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
