"""K-means clustering, written from scratch on top of numpy.

The two steps are kept as separate functions because they are separate ideas:
`assign_clusters` minimises the distortion over the assignments with the
centroids held fixed, and `move_centroids` minimises it over the centroids
with the assignments held fixed. Running them alternately is coordinate
descent on `distortion`, which is why the iteration converges and why it has
no step size to tune.
"""

import numpy as np

__all__ = ["assign_clusters", "move_centroids", "distortion", "init_centroids",
           "kmeans", "kmeans_best_of"]


def assign_clusters(X, centroids):
    """Index of the nearest centroid for every row of X.

    Args:
        X         (ndarray): (m, n) data.
        centroids (ndarray): (K, n) centroid positions.

    Returns:
        idx (ndarray): (m,) integer index in {0, ..., K-1}.
    """
    d2 = ((X[:, None, :] - centroids) ** 2).sum(axis=2)     # (m, K)
    return np.argmin(d2, axis=1)


def move_centroids(X, idx, K):
    """Move each centroid to the mean of the points assigned to it.

    A centroid that owns no points is left at the origin by this function;
    `kmeans` handles the empty-cluster case properly by keeping the previous
    position. Dividing by an empty count is the classic way this blows up.
    """
    n = X.shape[1]
    centroids = np.zeros((K, n))
    for k in range(K):
        members = X[idx == k]
        if len(members):
            centroids[k] = members.mean(axis=0)
    return centroids


def distortion(X, idx, centroids):
    """Mean squared distance from each point to its own centroid.

    Both steps of the algorithm decrease this, so it must never increase.
    Watching it is the most effective way to catch a bug in an implementation.
    """
    return float(np.mean(((X - centroids[idx]) ** 2).sum(axis=1)))


def init_centroids(X, K, rng):
    """Pick K distinct training examples as the starting centroids.

    Starting at real data points rather than at random locations guarantees
    every centroid begins somewhere the data actually are, so no cluster is
    empty on the first pass.
    """
    return X[rng.choice(len(X), K, replace=False)].copy()


def kmeans(X, K, rng=None, max_iters=100, tol=0.0, centroids=None):
    """Run K-means to convergence from one initialisation.

    Returns:
        centroids (ndarray): (K, n) final centroid positions.
        idx       (ndarray): (m,) final assignments.
        history   (ndarray): the distortion after every half-step, so
                             history[0::2] follows the assign steps and
                             history[1::2] the move steps. Non-increasing.
    """
    rng = np.random.default_rng() if rng is None else rng
    mu = init_centroids(X, K, rng) if centroids is None else centroids.copy()

    history = []
    idx = np.zeros(len(X), dtype=int)
    for _ in range(max_iters):
        idx = assign_clusters(X, mu)
        history.append(distortion(X, idx, mu))

        moved = move_centroids(X, idx, K)
        empty = np.array([not (idx == k).any() for k in range(K)])
        moved[empty] = mu[empty]          # keep an empty cluster where it was
        mu = moved
        history.append(distortion(X, idx, mu))

        if len(history) > 4 and abs(history[-1] - history[-3]) <= tol:
            break

    return mu, idx, np.array(history)


def kmeans_best_of(X, K, n_restarts=50, seed=0, max_iters=100):
    """Run K-means from many random initialisations and keep the best.

    This is not an optional refinement. K-means converges to a local minimum
    whose quality depends entirely on where it started, so a single run can
    land several times worse than the best available clustering.

    Returns:
        centroids, idx, best_distortion, all_distortions
    """
    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, n_restarts)
    best = None
    finals = []
    for s in seeds:
        mu, idx, hist = kmeans(X, K, np.random.default_rng(int(s)),
                               max_iters=max_iters)
        finals.append(hist[-1])
        if best is None or hist[-1] < best[2]:
            best = (mu, idx, hist[-1])
    return best[0], best[1], best[2], np.array(finals)
