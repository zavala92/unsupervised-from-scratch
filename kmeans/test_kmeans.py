#!/usr/bin/env python3
"""Tests for the K-means implementation. Run with:  python3 test_kmeans.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kmeans import (assign_clusters, move_centroids, distortion,
                    init_centroids, kmeans, kmeans_best_of)

TESTS = []


def test(name):
    def deco(fn):
        TESTS.append((name, fn))
        return fn
    return deco


def blobs(seed=0, per=40, spread=0.5):
    rng = np.random.default_rng(seed)
    centres = np.array([[0., 0.], [4., .4], [2., 3.6], [6., 3.4]])
    X = np.vstack([c + rng.standard_normal((per, 2)) * spread for c in centres])
    return X, centres


@test("assign_clusters picks the nearest centroid")
def _():
    X = np.array([[0., 0.], [10., 0.], [0., 10.]])
    C = np.array([[0., 0.], [10., 0.], [0., 10.]])
    assert list(assign_clusters(X, C)) == [0, 1, 2]
    # a point closer to centroid 1 than to 0
    assert assign_clusters(np.array([[6., 0.]]), C)[0] == 1


@test("move_centroids returns the per-cluster mean")
def _():
    X = np.array([[0., 0.], [2., 0.], [10., 10.], [12., 10.]])
    idx = np.array([0, 0, 1, 1])
    C = move_centroids(X, idx, 2)
    assert np.allclose(C, [[1., 0.], [11., 10.]]), C


@test("distortion matches an explicit loop")
def _():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((25, 3))
    C = rng.standard_normal((4, 3))
    idx = assign_clusters(X, C)
    want = np.mean([np.sum((X[i] - C[idx[i]]) ** 2) for i in range(len(X))])
    assert abs(distortion(X, idx, C) - want) < 1e-12


@test("init_centroids returns K distinct rows OF the data")
def _():
    X, _ = blobs()
    C = init_centroids(X, 4, np.random.default_rng(3))
    assert C.shape == (4, 2)
    assert len({tuple(r) for r in C}) == 4, "centroids must be distinct"
    for row in C:
        assert (X == row).all(axis=1).any(), "centroid is not a data point"


@test("the distortion never increases, at any half-step")
def _():
    X, _ = blobs()
    for seed in range(30):
        _, _, hist = kmeans(X, 4, np.random.default_rng(seed))
        rises = np.diff(hist)
        assert rises.max() <= 1e-12, (
            f"seed {seed}: distortion rose by {rises.max():.3e}. Both steps "
            "minimise the same objective, so this can only be a bug.")


@test("it converges, and the reported history ends at the reported state")
def _():
    X, _ = blobs()
    mu, idx, hist = kmeans(X, 4, np.random.default_rng(7))
    assert abs(distortion(X, idx, mu) - hist[-1]) < 1e-12
    assert np.array_equal(idx, assign_clusters(X, mu)), "assignments not stable"


@test("on well-separated blobs the best run recovers the true centres")
def _():
    X, centres = blobs(seed=5, spread=0.35)
    mu, _, J, _ = kmeans_best_of(X, 4, n_restarts=40, seed=0)
    # match each true centre to its closest recovered centroid
    for c in centres:
        assert np.min(np.linalg.norm(mu - c, axis=1)) < 0.25, mu


@test("restarts matter: some initialisations land in a worse local optimum")
def _():
    X, _ = blobs(seed=11, spread=0.52)
    _, _, best, finals = kmeans_best_of(X, 4, n_restarts=40, seed=0)
    assert finals.max() > best * 1.5, (
        "expected at least one restart to get stuck well above the best "
        f"value; got best {best:.3f}, worst {finals.max():.3f}")
    assert np.isclose(finals.min(), best)


@test("an empty cluster does not crash or produce nan")
def _():
    # K far larger than the number of distinct points forces empty clusters
    X = np.repeat(np.array([[0., 0.], [5., 5.]]), 6, axis=0)
    mu, idx, hist = kmeans(X, 5, np.random.default_rng(0))
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(hist))


@test("K = 1 puts the single centroid at the global mean")
def _():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((50, 3)) * 2 + 7
    mu, _, _ = kmeans(X, 1, np.random.default_rng(0))
    assert np.allclose(mu[0], X.mean(axis=0), atol=1e-10)


@test("distortion decreases monotonically in K")
def _():
    X, _ = blobs(seed=4)
    prev = np.inf
    for K in range(1, 7):
        _, _, J, _ = kmeans_best_of(X, K, n_restarts=20, seed=1)
        assert J <= prev + 1e-9, f"J rose from {prev:.4f} to {J:.4f} at K={K}"
        prev = J


def main():
    print("\nK-means test suite")
    print("=" * 66)
    fails = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}\n        {e}")
            fails += 1
    print("=" * 66)
    print(f"{len(TESTS) - fails} passed, {fails} failed\n")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
