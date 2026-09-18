"""Reference implementation of the anomaly-detection core.

Do not read this until you have made a serious attempt at anomdet/core.py.
(Or do, but read it actively: cover the body, try to reproduce it, peek only
for a hint.  That is how you get the most out of it.)
"""

import numpy as np

_LOG_2PI = np.log(2.0 * np.pi)


def fit_independent(X):
    m, n = X.shape
    mu = X.sum(axis=0) / m
    var = ((X - mu) ** 2).sum(axis=0) / m
    return mu, var


def independent_pdf(X, mu, var):
    X = np.atleast_2d(X)
    z = (X - mu) ** 2 / var
    factors = np.exp(-0.5 * z) / np.sqrt(2.0 * np.pi * var)
    return np.prod(factors, axis=1)


def log_independent_pdf(X, mu, var):
    X = np.atleast_2d(X)
    z = (X - mu) ** 2 / var
    terms = -0.5 * (z + np.log(var) + _LOG_2PI)
    return terms.sum(axis=1)


def precision_recall_f1(y_true, y_pred):
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)

    tp = int(np.sum(y_pred & y_true))
    fp = int(np.sum(y_pred & ~y_true))
    fn = int(np.sum(~y_pred & y_true))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def tune_threshold(y_val, p_val, n_steps=1000):
    p_val = np.asarray(p_val, dtype=float)

    best_eps, best_f1 = 0.0, 0.0
    lo, hi = p_val.min(), p_val.max()
    step = (hi - lo) / n_steps
    if step <= 0.0:
        return best_eps, best_f1

    for eps in np.arange(lo, hi + step, step):
        predictions = p_val < eps
        _, _, f1 = precision_recall_f1(y_val, predictions)
        if f1 > best_f1:
            best_f1, best_eps = f1, float(eps)
    return best_eps, best_f1


def tune_threshold_log(y_val, logp_val, n_steps=1000):
    logp_val = np.asarray(logp_val, dtype=float)

    best_t, best_f1 = -np.inf, 0.0
    lo, hi = logp_val.min(), logp_val.max()
    step = (hi - lo) / n_steps
    if step <= 0.0:
        return best_t, best_f1

    for t in np.arange(lo, hi + step, step):
        predictions = logp_val < t
        _, _, f1 = precision_recall_f1(y_val, predictions)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def correlated_pdf(X, mu, Sigma):
    X = np.atleast_2d(X)
    n = mu.shape[0]
    d = X - mu

    # Cholesky: Sigma = L L^T.  Solving L w = d^T gives w^T w = d^T Sigma^{-1} d
    # and log det Sigma = 2 sum log diag(L), both without ever forming Sigma^{-1}.
    L = np.linalg.cholesky(Sigma)
    w = np.linalg.solve(L, d.T)
    maha = np.sum(w * w, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))

    logp = -0.5 * (maha + log_det + n * _LOG_2PI)
    return np.exp(logp)


def log_correlated_pdf(X, mu, Sigma):
    X = np.atleast_2d(X)
    n = mu.shape[0]
    d = X - mu
    L = np.linalg.cholesky(Sigma)
    w = np.linalg.solve(L, d.T)
    maha = np.sum(w * w, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (maha + log_det + n * _LOG_2PI)


def fit_correlated(X):
    m, n = X.shape
    mu = X.sum(axis=0) / m
    d = X - mu
    Sigma = (d.T @ d) / m
    return mu, Sigma
