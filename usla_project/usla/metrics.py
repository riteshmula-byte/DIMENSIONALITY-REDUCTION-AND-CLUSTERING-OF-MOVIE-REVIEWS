"""Embedding-quality metrics that stay cheap at n = 25,000 (sampled queries)."""
import numpy as np
from sklearn.metrics import pairwise_distances


def neighbourhood_metrics(Z, Y, labels, k=10, n_queries=2000, seed=0, chunk=250):
    """Sampled trustworthiness, kNN preservation and label agreement.

    For each query point the k nearest neighbours are found against *all* points
    in the original SVD space (Z) and in the 2-D map (Y).
    With n_queries >= n this equals sklearn's exact trustworthiness.
    """
    n = len(Z)
    q = np.arange(n) if n <= n_queries else np.sort(
        np.random.RandomState(seed).choice(n, n_queries, replace=False))
    pen = ov = lab_y = lab_x = 0.0
    for s in range(0, len(q), chunk):
        idx = q[s:s + chunk]
        rows = np.arange(len(idx))
        dX = pairwise_distances(Z[idx], Z, squared=True)
        dY = pairwise_distances(Y[idx], Y, squared=True)
        dX[rows, idx] = np.inf
        dY[rows, idx] = np.inf
        ordX = np.argsort(dX, axis=1)
        inv = np.empty_like(ordX)
        inv[rows[:, None], ordX] = np.arange(1, n + 1)[None, :]
        nnX = ordX[:, :k]
        nnY = np.argpartition(dY, k, axis=1)[:, :k]
        r = inv[rows[:, None], nnY]
        pen += np.maximum(r - k, 0).sum()
        ov += (r <= k).sum() / k
        lab_y += (labels[nnY] == labels[idx][:, None]).mean(axis=1).sum()
        lab_x += (labels[nnX] == labels[idx][:, None]).mean(axis=1).sum()
    m = len(q)
    return {"trustworthiness": 1 - 2 * pen / (m * k * (2 * n - 3 * k - 1)),
            "knn_preservation": ov / m,
            "label_agreement_2d": lab_y / m,
            "label_agreement_hd": lab_x / m}
