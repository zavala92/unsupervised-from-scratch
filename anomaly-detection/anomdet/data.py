"""Synthetic datasets for the stages.  All deterministic given a seed, all
generated offline, nothing to download.

Design rule: the TRAINING set contains only normal examples (that is the whole
premise of anomaly detection, anomalies are too rare to collect).  The
VALIDATION and TEST sets contain a few labelled anomalies, which is what lets
you tune and then honestly score the threshold.
"""

from collections import namedtuple

import numpy as np

Dataset = namedtuple(
    "Dataset",
    "X_train X_val y_val X_test y_test feature_names meta",
    defaults=(None,),
)


# --------------------------------------------------------------------- helpers

def _mahalanobis(X, mu, Sigma):
    L = np.linalg.cholesky(Sigma)
    w = np.linalg.solve(L, (X - mu).T)
    return np.sqrt(np.sum(w * w, axis=0))


def _sample_far_outliers(rng, k, mu, Sigma, lo, hi, d_min, d_max=np.inf,
                         max_zscore=np.inf):
    """Rejection-sample k points in the box [lo, hi] whose Mahalanobis distance
    from N(mu, Sigma) lies in [d_min, d_max], and whose per-coordinate z-score
    is at most max_zscore.

    That last constraint is the interesting one: it produces points that are
    unremarkable in every individual feature yet jointly impossible.  Those are
    exactly the anomalies an axis-aligned model cannot see (stage 5).
    """
    mu, lo, hi = np.asarray(mu), np.asarray(lo), np.asarray(hi)
    sd = np.sqrt(np.diag(Sigma))
    out = []
    while len(out) < k:
        cand = rng.uniform(lo, hi, size=(64, mu.shape[0]))
        d = _mahalanobis(cand, mu, Sigma)
        z = np.max(np.abs(cand - mu) / sd, axis=1)
        keep = (d >= d_min) & (d <= d_max) & (z <= max_zscore)
        for row in cand[keep]:
            out.append(row)
            if len(out) == k:
                break
    return np.array(out[:k])


def _shuffle_together(rng, X, y):
    idx = rng.permutation(X.shape[0])
    return X[idx], y[idx]


# ------------------------------------------------------------- STAGE 1 dataset

def engine_heat_1d(seed=0, m_train=400, n_val=150, n_anom=10):
    """One feature only (engine heat), so you can plot p(x) as an actual curve
    and check by eye that the fitted density does what you expect."""
    rng = np.random.default_rng(seed)
    mu_true, sd_true = 70.0, 4.0

    X_train = rng.normal(mu_true, sd_true, size=(m_train, 1))

    normal_val = rng.normal(mu_true, sd_true, size=(n_val, 1))
    anom_val = np.concatenate([
        rng.uniform(46.0, 54.0, size=(n_anom // 2, 1)),
        rng.uniform(86.0, 96.0, size=(n_anom - n_anom // 2, 1)),
    ])
    X_val = np.vstack([normal_val, anom_val])
    y_val = np.concatenate([np.zeros(n_val), np.ones(n_anom)])
    X_val, y_val = _shuffle_together(rng, X_val, y_val)

    normal_te = rng.normal(mu_true, sd_true, size=(n_val, 1))
    anom_te = np.concatenate([
        rng.uniform(46.0, 54.0, size=(n_anom // 2, 1)),
        rng.uniform(86.0, 96.0, size=(n_anom - n_anom // 2, 1)),
    ])
    X_test = np.vstack([normal_te, anom_te])
    y_test = np.concatenate([np.zeros(n_val), np.ones(n_anom)])
    X_test, y_test = _shuffle_together(rng, X_test, y_test)

    return Dataset(X_train, X_val, y_val, X_test, y_test,
                   ["heat [C]"], {"mu_true": mu_true, "sd_true": sd_true})


# ----------------------------------------------------- STAGE 2 & 3 dataset

def engine_data(seed=1, m_train=400, n_val=160, n_anom=12):
    """The lecture's example: heat vs vibration, mildly correlated, with
    outliers scattered well away from the cloud."""
    rng = np.random.default_rng(seed)
    mu = np.array([70.0, 10.0])
    Sigma = np.array([[16.0, 3.2],
                      [3.2, 2.25]])
    L = np.linalg.cholesky(Sigma)

    def draw(k):
        return mu + rng.standard_normal((k, 2)) @ L.T

    lo, hi = np.array([48.0, 3.0]), np.array([95.0, 18.0])

    X_train = draw(m_train)

    X_val = np.vstack([draw(n_val),
                       _sample_far_outliers(rng, n_anom, mu, Sigma, lo, hi,
                                            d_min=4.2)])
    y_val = np.concatenate([np.zeros(n_val), np.ones(n_anom)])
    X_val, y_val = _shuffle_together(rng, X_val, y_val)

    X_test = np.vstack([draw(n_val),
                        _sample_far_outliers(rng, n_anom, mu, Sigma, lo, hi,
                                             d_min=4.2)])
    y_test = np.concatenate([np.zeros(n_val), np.ones(n_anom)])
    X_test, y_test = _shuffle_together(rng, X_test, y_test)

    return Dataset(X_train, X_val, y_val, X_test, y_test,
                   ["heat [C]", "vibration [mm/s]"],
                   {"mu_true": mu, "Sigma_true": Sigma})


# ------------------------------------------------------------- STAGE 4 dataset

def skewed_server_data(seed=7, m_train=600, n_val=300, n_anom=15):
    """Two features with very different shapes:

        latency:    log-normal, a long but entirely LEGITIMATE right tail.
                      It carries NO anomaly signal.  It is a nuisance feature.
        throughput: comfortably Gaussian, and where the real signal lives:
                      anomalous servers are starved, ~4.5 sigma low.

    Fit a Gaussian straight onto `latency` and its variance inflates to cover
    the tail, so an honest slow-but-fine request lands 4 sigma out and scores
    as MORE anomalous than a genuinely starved server.  The nuisance feature
    drowns the signal and precision collapses.  Take log(latency) first --
    which makes that feature exactly Gaussian, and the false positives go
    away while recall is untouched.

    That asymmetry is the thing to notice: bad feature shape costs you
    PRECISION, not recall.
    """
    rng = np.random.default_rng(seed)
    log_med, sigma_log = np.log(120.0), 0.8
    tp_mu, tp_sd = 800.0, 90.0
    drop = 4.5

    def draw_normal(k):
        return np.column_stack([np.exp(rng.normal(log_med, sigma_log, k)),
                                rng.normal(tp_mu, tp_sd, k)])

    def draw_anom(k):
        # Latency drawn from the SAME distribution as normal traffic: the
        # anomaly is invisible in that coordinate, by construction.
        return np.column_stack([np.exp(rng.normal(log_med, sigma_log, k)),
                                rng.normal(tp_mu - drop * tp_sd, 0.35 * tp_sd, k)])

    def split(n_norm, n_a):
        X = np.vstack([draw_normal(n_norm), draw_anom(n_a)])
        y = np.concatenate([np.zeros(n_norm), np.ones(n_a)])
        return _shuffle_together(rng, X, y)

    X_train = draw_normal(m_train)
    X_val, y_val = split(n_val, n_anom)
    X_test, y_test = split(n_val, n_anom)

    return Dataset(X_train, X_val, y_val, X_test, y_test,
                   ["latency [ms]", "throughput [req/s]"],
                   {"log_med": log_med, "sigma_log": sigma_log,
                    "skewed_columns": [0]})


def high_dim_normal(seed=3, m=200, n=800):
    """Standard normal in n dimensions: fuel for the underflow experiment."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, n))


# ------------------------------------------------------------- STAGE 5 dataset

def correlated_data(seed=4, m_train=500, n_val=200, n_anom=14):
    """A strongly correlated cloud (rho = 0.92) plus anomalies chosen so that
    EVERY individual coordinate is within 2.1 standard deviations of its mean,
    while the joint Mahalanobis distance exceeds 4.

    The axis-aligned model cannot flag these at any threshold without also
    flagging a pile of normal points.  The full-covariance model finds them
    immediately.  That gap is the entire point of the stage.
    """
    rng = np.random.default_rng(seed)
    mu = np.array([0.0, 0.0])
    rho = 0.92
    Sigma = np.array([[1.0, rho],
                      [rho, 1.0]])
    L = np.linalg.cholesky(Sigma)

    def draw(k):
        return mu + rng.standard_normal((k, 2)) @ L.T

    lo, hi = np.array([-2.6, -2.6]), np.array([2.6, 2.6])

    X_train = draw(m_train)

    X_val = np.vstack([draw(n_val),
                       _sample_far_outliers(rng, n_anom, mu, Sigma, lo, hi,
                                            d_min=4.0, max_zscore=2.1)])
    y_val = np.concatenate([np.zeros(n_val), np.ones(n_anom)])
    X_val, y_val = _shuffle_together(rng, X_val, y_val)

    X_test = np.vstack([draw(n_val),
                        _sample_far_outliers(rng, n_anom, mu, Sigma, lo, hi,
                                             d_min=4.0, max_zscore=2.1)])
    y_test = np.concatenate([np.zeros(n_val), np.ones(n_anom)])
    X_test, y_test = _shuffle_together(rng, X_test, y_test)

    return Dataset(X_train, X_val, y_val, X_test, y_test,
                   ["x1", "x2"], {"mu_true": mu, "Sigma_true": Sigma})


# ------------------------------------------------------------- STAGE 6 capstone

SOLVER_FEATURES = ["cond_est", "iters", "res_final", "rate_per_iter",
                   "setup_s", "mem_gb"]

#: What each planted failure mode is, and which lesson you need to catch it.
SOLVER_FAILURE_MODES = {
    1: ("stagnation",
        "Krylov hits the iteration cap, residual barely moves. Loud in "
        "several features at once, any working detector should get this."),
    2: ("preconditioner degradation",
        "Iteration count is normal-ish ON ITS OWN and the condition number is "
        "normal-ish ON ITS OWN, but the run needs roughly twice the iterations "
        "its conditioning predicts. Only a model that knows the two features "
        "are correlated can see it, this is stage 5's lesson."),
    3: ("silently easier problem",
        "The operator came out far better conditioned than the mesh warrants "
        "(a coefficient never got applied), and it converged suspiciously "
        "fast. cond_est is wildly heavy-tailed, so an untransformed Gaussian "
        "puts a LOW cond_est well under 1 sigma from its own inflated mean and "
        "sees nothing, this is stage 4's lesson."),
}


def solver_runs(seed=11, m_train=900, n_val=400, n_anom_each=8,
                n_production=300):
    """Telemetry from a (synthetic) iterative linear solver: the capstone.

    Six observable features per run::

        cond_est       estimated condition number      (violently heavy-tailed)
        iters          Krylov iterations to tolerance  (tracks log cond_est)
        res_final      final relative residual         (heavy-tailed)
        rate_per_iter  mean log10 residual drop / iter
        setup_s        preconditioner setup time       (tracks problem size)
        mem_gb         peak memory                     (tracks problem size)

    Problem size and conditioning are LATENT, they are not columns, but they
    induce real correlation between the columns you do get.  That is what makes
    the axis-aligned model leave something on the table.

    Three failure modes are planted in the labelled splits, in equal numbers,
    each needing a different tool to catch (see SOLVER_FAILURE_MODES).  The
    validation and test sets carry `modes_val` / `modes_test` in `meta`: 0 for
    a normal run, 1/2/3 for the mode.  Score recall per mode; an aggregate F1
    hides which lesson you have actually absorbed.

    `meta["X_production"]` is an UNLABELLED batch.  Once your detector is
    tuned, flag it and report how many runs you would send for inspection.
    """
    rng = np.random.default_rng(seed)

    def normal_runs(k):
        s = rng.uniform(4.0, 6.0, k)                       # latent log10(dofs)
        kap = rng.normal(6.0, 0.8, k)                      # latent log10(cond)
        cond = 10.0 ** kap
        iters = 25.0 + 6.0 * (kap - 6.0) + rng.normal(0, 2.5, k)
        iters = np.maximum(iters, 4.0)
        res = 10.0 ** rng.normal(-10.0, 0.45, k)
        rate = -np.log10(res) / iters
        setup = 0.02 * 10.0 ** (1.05 * (s - 4.0)) * np.exp(rng.normal(0, 0.18, k))
        mem = 0.40 * 10.0 ** (s - 4.0) * np.exp(rng.normal(0, 0.15, k))
        return np.column_stack([cond, iters, res, rate, setup, mem]), s, kap

    def mode1(k):  # stagnation
        X, s, kap = normal_runs(k)
        X[:, 1] = 200.0 + rng.normal(0, 3.0, k)            # iteration cap
        X[:, 2] = 10.0 ** rng.normal(-3.0, 0.3, k)         # residual stuck
        X[:, 3] = -np.log10(X[:, 2]) / X[:, 1]
        return X

    def mode2(k):  # preconditioner degradation: breaks iters ~ log cond
        X, s, kap = normal_runs(k)
        kap_low = rng.normal(5.0, 0.25, k)                 # benign conditioning
        X[:, 0] = 10.0 ** kap_low
        expected = 25.0 + 6.0 * (kap_low - 6.0)            # ~19 iterations
        X[:, 1] = 2.05 * expected + rng.normal(0, 1.5, k)  # ~39: marginally dull
        X[:, 3] = -np.log10(X[:, 2]) / X[:, 1]
        return X

    def mode3(k):  # silently easier problem
        X, s, kap = normal_runs(k)
        kap_easy = rng.normal(6.0 - 3.2 * 0.8, 0.18, k)    # 3.2 sigma low in LOG
        X[:, 0] = 10.0 ** kap_easy
        X[:, 1] = np.maximum(25.0 + 6.0 * (kap_easy - 6.0) + rng.normal(0, 1.2, k), 3.0)
        X[:, 3] = -np.log10(X[:, 2]) / X[:, 1]
        return X

    def split(n_norm, n_each):
        Xn, _, _ = normal_runs(n_norm)
        blocks = [Xn], [np.zeros(n_norm, dtype=int)]
        for tag, fn in ((1, mode1), (2, mode2), (3, mode3)):
            blocks[0].append(fn(n_each))
            blocks[1].append(np.full(n_each, tag, dtype=int))
        X = np.vstack(blocks[0])
        modes = np.concatenate(blocks[1])
        idx = rng.permutation(X.shape[0])
        X, modes = X[idx], modes[idx]
        return X, (modes > 0).astype(float), modes

    X_train, _, _ = normal_runs(m_train)
    X_val, y_val, modes_val = split(n_val, n_anom_each)
    X_test, y_test, modes_test = split(n_val, n_anom_each)

    # Unlabelled production batch: mostly healthy with a light sprinkle.
    Xp, _, _ = normal_runs(n_production - 9)
    X_production = np.vstack([Xp, mode1(3), mode2(3), mode3(3)])
    X_production = X_production[rng.permutation(X_production.shape[0])]

    return Dataset(X_train, X_val, y_val, X_test, y_test, SOLVER_FEATURES,
                   {"modes_val": modes_val, "modes_test": modes_test,
                    "X_production": X_production,
                    "skewed_columns": [0, 2, 4, 5],
                    "failure_modes": SOLVER_FAILURE_MODES})
