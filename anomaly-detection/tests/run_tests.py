#!/usr/bin/env python3
"""Test runner for the anomaly-detection project.  No pytest needed.

    python3 tests/run_tests.py              # test YOUR code (anomdet/core.py)
    python3 tests/run_tests.py --stage 3    # only the stage-3 tests
    ANOMDET_USE_REFERENCE=1 python3 tests/run_tests.py    # sanity-check the reference

A function you have not written yet reports as TODO, not as a failure, so the
summary line doubles as a progress bar through the README stages.
"""

import argparse
import inspect
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anomdet
from anomdet import core

TESTS = []


def test(stage, name):
    def deco(fn):
        TESTS.append((stage, name, fn))
        return fn
    return deco


def close(a, b, tol=1e-12, msg=""):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise AssertionError(f"shape {a.shape} != {b.shape}. {msg}")
    err = np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b)))
    if not err <= tol:
        raise AssertionError(f"max rel err {err:.3e} > {tol:.1e}. {msg}\n"
                             f"  got      {a.ravel()[:6]}\n  expected {b.ravel()[:6]}")


def _code_without_docstring(fn):
    """Source of fn with the docstring removed, so scanning the CODE for banned
    calls is not confused by prose in the docstring that mentions them."""
    import ast
    import textwrap
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, SyntaxError):
        return None
    node = tree.body[0]
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        node.body = node.body[1:]
    if not node.body:
        return None
    return ast.unparse(node)


def gauss_legendre(f, a, b, n=200):
    x, w = np.polynomial.legendre.leggauss(n)
    xm = 0.5 * (b - a) * x + 0.5 * (a + b)
    return 0.5 * (b - a) * np.sum(w * f(xm))


# ============================================================ STAGE 1 & 2

@test(1, "fit_independent matches a hand-computed mean and variance")
def _():
    X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0], [7.0, 40.0]])
    mu, var = anomdet.fit_independent(X)
    close(mu, [4.0, 25.0], msg="mean")
    # deviations 3,1,1,3 -> mean square 5 ; and 15,5,5,15 -> 125
    close(var, [5.0, 125.0], msg="ML variance (divide by m, not m-1)")


@test(1, "fit_independent uses the ML normalisation 1/m")
def _():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((37, 4))
    _, var = anomdet.fit_independent(X)
    close(var, X.var(axis=0, ddof=0), tol=1e-13,
          msg="looks like you divided by m-1")


@test(1, "independent_pdf reproduces the analytic 1-D density")
def _():
    mu, var = np.array([2.0]), np.array([9.0])
    x = np.array([[2.0], [5.0], [-4.0]])
    want = np.exp(-0.5 * ((x[:, 0] - 2.0) / 3.0) ** 2) / (3.0 * np.sqrt(2 * np.pi))
    close(anomdet.independent_pdf(x, mu, var), want, tol=1e-13)


@test(1, "independent_pdf integrates to 1 in 1-D (Gauss-Legendre)")
def _():
    mu, var = np.array([1.7]), np.array([2.3])
    sd = np.sqrt(var[0])
    I = gauss_legendre(lambda t: anomdet.independent_pdf(t[:, None], mu, var),
                       mu[0] - 12 * sd, mu[0] + 12 * sd, n=400)
    close(I, 1.0, tol=1e-10, msg="density is not normalised")


@test(2, "independent_pdf integrates to 1 in 2-D")
def _():
    mu, var = np.array([0.5, -1.0]), np.array([4.0, 0.25])
    sd = np.sqrt(var)
    nq = 160
    xq, wq = np.polynomial.legendre.leggauss(nq)
    tot = 0.0
    for j, (m0, s0) in enumerate(zip(mu, sd)):
        pass
    a, b = mu - 10 * sd, mu + 10 * sd
    X0 = 0.5 * (b[0] - a[0]) * xq + 0.5 * (a[0] + b[0])
    X1 = 0.5 * (b[1] - a[1]) * xq + 0.5 * (a[1] + b[1])
    G0, G1 = np.meshgrid(X0, X1, indexing="ij")
    W = np.outer(wq, wq) * 0.25 * (b[0] - a[0]) * (b[1] - a[1])
    P = anomdet.independent_pdf(np.column_stack([G0.ravel(), G1.ravel()]), mu, var)
    close(np.sum(W.ravel() * P), 1.0, tol=1e-10)


@test(2, "independent_pdf accepts a single point of shape (n,)")
def _():
    mu, var = np.array([0.0, 1.0]), np.array([1.0, 4.0])
    one = anomdet.independent_pdf(np.array([0.3, 1.1]), mu, var)
    many = anomdet.independent_pdf(np.array([[0.3, 1.1]]), mu, var)
    assert np.shape(one) == (1,), f"expected shape (1,), got {np.shape(one)}"
    close(one, many)


@test(2, "independent_pdf factorises over features, as an independent model must")
def _():
    rng = np.random.default_rng(3)
    mu, var = rng.standard_normal(5), rng.uniform(.5, 3., 5)
    X = rng.standard_normal((11, 5))
    joint = anomdet.independent_pdf(X, mu, var)
    marg = np.ones(11)
    for j in range(5):
        marg *= anomdet.independent_pdf(X[:, [j]], mu[[j]], var[[j]])
    close(joint, marg, tol=1e-12)


@test(2, "log_independent_pdf agrees with log(independent_pdf)")
def _():
    rng = np.random.default_rng(4)
    mu, var = rng.standard_normal(3), rng.uniform(.4, 2., 3)
    X = mu + rng.standard_normal((50, 3)) * np.sqrt(var)
    close(anomdet.log_independent_pdf(X, mu, var),
          np.log(anomdet.independent_pdf(X, mu, var)), tol=1e-11)


@test(2, "log_independent_pdf still works where independent_pdf underflows to zero")
def _():
    rng = np.random.default_rng(5)
    n = 800
    mu, var = np.zeros(n), np.ones(n)
    X = rng.standard_normal((6, n))
    p = anomdet.independent_pdf(X, mu, var)
    logp = anomdet.log_independent_pdf(X, mu, var)
    assert np.all(p == 0.0), ("this test assumes p underflows at n=800; it did "
                              "not, so something is off")
    assert np.all(np.isfinite(logp)) and np.all(logp < -100), (
        "log_independent_pdf must not go through p(x); it has to sum logs. "
        f"got {logp}")


# ================================================================= STAGE 3

@test(3, "precision_recall_f1 on a hand-built confusion matrix")
def _():
    #                 tp=2, fp=1, fn=2, tn=3
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    p, r, f1 = anomdet.precision_recall_f1(y_true, y_pred)
    close(p, 2 / 3, tol=1e-14, msg="precision")
    close(r, 0.5, tol=1e-14, msg="recall")
    close(f1, 2 * (2 / 3) * .5 / ((2 / 3) + .5), tol=1e-14, msg="F1")


@test(3, "precision_recall_f1 returns 0, never nan, on the degenerate cases")
def _():
    y = np.array([1, 0, 0, 1])
    for name, yt, yp in [("flagged nothing", y, np.zeros(4)),
                         ("no true anomalies", np.zeros(4), np.array([1, 0, 0, 1])),
                         ("both empty", np.zeros(4), np.zeros(4))]:
        out = anomdet.precision_recall_f1(yt, yp)
        assert all(np.isfinite(v) for v in out), f"{name}: got {out}"
        assert out[2] == 0.0, f"{name}: F1 should be 0.0, got {out[2]}"


@test(3, "precision_recall_f1 treats bool and 0/1 input identically")
def _():
    yt = np.array([1, 1, 0, 0, 1])
    yp = np.array([1, 0, 0, 1, 1])
    close(anomdet.precision_recall_f1(yt, yp),
          anomdet.precision_recall_f1(yt.astype(bool), yp.astype(bool)))


@test(3, "tune_threshold finds F1 = 1 when the classes are separable")
def _():
    p_val = np.concatenate([np.full(90, 0.4), np.full(10, 1e-6)])
    y_val = np.concatenate([np.zeros(90), np.ones(10)])
    eps, f1 = anomdet.tune_threshold(y_val, p_val)
    close(f1, 1.0, tol=1e-12, msg="should be perfectly separable")
    assert 1e-6 < eps <= 0.4, f"epsilon {eps:g} does not separate the two groups"


@test(3, "tune_threshold never reports F1 outside [0, 1]")
def _():
    rng = np.random.default_rng(6)
    for _ in range(5):
        p_val = rng.uniform(0, 1, 200)
        y_val = (rng.uniform(0, 1, 200) < .1).astype(float)
        eps, f1 = anomdet.tune_threshold(y_val, p_val)
        assert 0.0 <= f1 <= 1.0 + 1e-15, f1


@test(3, "tune_threshold_log picks an equivalent cut to tune_threshold")
def _():
    rng = np.random.default_rng(7)
    p_val = np.concatenate([rng.uniform(.2, .9, 150), rng.uniform(1e-9, 1e-7, 12)])
    y_val = np.concatenate([np.zeros(150), np.ones(12)])
    _, f1_lin = anomdet.tune_threshold(y_val, p_val)
    t, f1_log = anomdet.tune_threshold_log(y_val, np.log(p_val))
    close(f1_log, f1_lin, tol=1e-12, msg="the two searches disagree")
    pred_log = np.log(p_val) < t
    assert anomdet.precision_recall_f1(y_val, pred_log)[2] == f1_log


# ================================================================= STAGE 5

@test(5, "fit_correlated matches np.cov with the ML normalisation")
def _():
    rng = np.random.default_rng(8)
    X = rng.standard_normal((60, 3)) @ np.array([[2., .5, 0], [0, 1., .3], [0, 0, .7]])
    mu, S = anomdet.fit_correlated(X)
    close(mu, X.mean(axis=0), tol=1e-13)
    close(S, np.cov(X, rowvar=False, bias=True), tol=1e-12)
    close(S, S.T, tol=1e-14, msg="covariance must be symmetric")


@test(5, "a diagonal Sigma reproduces the independent model exactly")
def _():
    rng = np.random.default_rng(9)
    mu, var = rng.standard_normal(4), rng.uniform(.5, 2.5, 4)
    X = mu + rng.standard_normal((25, 4))
    close(anomdet.log_correlated_pdf(X, mu, np.diag(var)),
          anomdet.log_independent_pdf(X, mu, var), tol=1e-11)


@test(5, "the full-covariance density integrates to 1 in 2-D")
def _():
    mu = np.array([.3, -.4])
    S = np.array([[2.0, 1.2], [1.2, 1.0]])
    nq = 200
    xq, wq = np.polynomial.legendre.leggauss(nq)
    sd = np.sqrt(np.diag(S))
    a, b = mu - 12 * sd, mu + 12 * sd
    X0 = .5 * (b[0] - a[0]) * xq + .5 * (a[0] + b[0])
    X1 = .5 * (b[1] - a[1]) * xq + .5 * (a[1] + b[1])
    G0, G1 = np.meshgrid(X0, X1, indexing="ij")
    W = np.outer(wq, wq) * .25 * (b[0] - a[0]) * (b[1] - a[1])
    P = anomdet.correlated_pdf(np.column_stack([G0.ravel(), G1.ravel()]), mu, S)
    close(np.sum(W.ravel() * P), 1.0, tol=1e-9)


@test(5, "the full-covariance density is invariant under rotation")
def _():
    rng = np.random.default_rng(10)
    th = 0.7
    Q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    mu = np.array([.2, -.1])
    S = np.array([[3.0, 1.1], [1.1, .8]])
    X = rng.standard_normal((30, 2))
    close(anomdet.log_correlated_pdf(X @ Q.T, Q @ mu, Q @ S @ Q.T),
          anomdet.log_correlated_pdf(X, mu, S), tol=1e-11,
          msg="p(Qx; Qmu, QSQ^T) must equal p(x; mu, S)")


@test(5, "correlated_pdf is exp of its own log")
def _():
    rng = np.random.default_rng(11)
    mu = rng.standard_normal(3)
    A = rng.standard_normal((3, 3))
    S = A @ A.T + 3 * np.eye(3)
    X = mu + rng.standard_normal((20, 3))
    close(anomdet.correlated_pdf(X, mu, S),
          np.exp(anomdet.log_correlated_pdf(X, mu, S)), tol=1e-12)


@test(5, "log_correlated_pdf avoids np.linalg.inv and np.linalg.det")
def _():
    # Touch it first so an unwritten function reports TODO rather than FAIL.
    anomdet.log_correlated_pdf(np.zeros((1, 2)), np.zeros(2), np.eye(2))
    src = _code_without_docstring(core.log_correlated_pdf)
    if src is None:
        return
    for banned in ("linalg.inv", "linalg.det", "linalg.slogdet", "np.inv("):
        assert banned not in src, (
            f"found `{banned}`. Use the Cholesky factor instead: it is cheaper, "
            "better conditioned, and gives you log det for free.")
    assert "cholesky" in src.lower(), "expected a Cholesky factorisation here"


# ==================================================================== runner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=None,
                    help="run only the tests for this stage")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show full tracebacks for failures")
    args = ap.parse_args()

    selected = [t for t in TESTS if args.stage is None or t[0] == args.stage]
    mode = "REFERENCE SOLUTION" if anomdet.USING_SOLUTION else "your anomdet/core.py"
    print(f"\nanomaly-detection test suite  [{mode}]")
    print("=" * 74)

    npass = nfail = ntodo = 0
    cur = None
    for stage, name, fn in selected:
        if stage != cur:
            cur = stage
            print(f"\n-- stage {stage} " + "-" * (68 - len(str(stage))))
        try:
            fn()
            print(f"  PASS  {name}")
            npass += 1
        except NotImplementedError as e:
            print(f"  TODO  {name}   ({e} not written yet)")
            ntodo += 1
        except AssertionError as e:
            print(f"  FAIL  {name}\n        {e}")
            nfail += 1
        except Exception as e:
            print(f"  FAIL  {name}\n        {type(e).__name__}: {e}")
            if args.verbose:
                traceback.print_exc()
            nfail += 1

    print("\n" + "=" * 74)
    print(f"{npass} passed, {nfail} failed, {ntodo} not yet implemented "
          f"({npass}/{len(selected)} of the suite)")
    if ntodo and not nfail:
        print("Keep going; nothing you have written is broken.")
    elif nfail == 0 and ntodo == 0:
        print("All green. Run the stages and look at the figures.")
    print()
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
