"""Exact PAM (BUILD + steepest SWAP) evaluated in column blocks.

Same algorithm as the original notebook, but the n x n distance matrix is never
stored, so it scales to the full 25,000 reviews in a few hundred MB.
"""
import numpy as np


class _Dist:
    def __init__(self, Z, dtype):
        self.Z = Z.astype(dtype)
        self.sq = np.einsum("ij,ij->i", self.Z, self.Z)

    def cols(self, idx):
        d = self.sq[:, None] + self.sq[idx][None, :] - 2 * (self.Z @ self.Z[idx].T)
        return np.sqrt(np.maximum(d, 0))


def k_medoids(Z, n_clusters, max_iter=100, block=512):
    n = len(Z)
    if not 1 <= n_clusters < n:
        raise ValueError("n_clusters must be in [1, n-1]")
    D = _Dist(Z, np.float64 if n <= 5000 else np.float32)
    blocks = [np.arange(s, min(s + block, n)) for s in range(0, n, block)]

    rowsum = np.concatenate([D.cols(b).sum(axis=0) for b in blocks])
    medoids = [int(np.argmin(rowsum))]
    nearest = D.cols(np.array(medoids))[:, 0]
    while len(medoids) < n_clusters:                      # BUILD
        gains = np.concatenate([np.maximum(nearest[:, None] - D.cols(b), 0).sum(axis=0)
                                for b in blocks])
        gains[medoids] = -np.inf
        medoids.append(int(np.argmax(gains)))
        nearest = np.minimum(nearest, D.cols(np.array(medoids[-1:]))[:, 0])
    medoids = np.array(medoids)

    dm = D.cols(medoids)
    cost = float(dm.min(axis=1).sum())
    history, swaps = [cost], 0
    for _ in range(max_iter):                             # SWAP
        dm = D.cols(medoids)
        excl = []
        for j in range(n_clusters):
            others = np.delete(dm, j, axis=1)
            excl.append(others.min(axis=1) if others.shape[1] else np.full(n, np.inf, dm.dtype))
        best = (cost, None, None)
        for b in blocks:
            Db = D.cols(b)
            for j in range(n_clusters):
                c = np.minimum(excl[j][:, None], Db).sum(axis=0).astype(np.float64)
                c[np.isin(b, medoids)] = np.inf
                i = int(np.argmin(c))
                if c[i] < best[0] - 1e-9:
                    best = (float(c[i]), j, int(b[i]))
        if best[1] is None:
            break
        medoids[best[1]] = best[2]
        cost = best[0]
        swaps += 1
        history.append(cost)
    dm = D.cols(medoids)
    labels = np.argmin(dm, axis=1)
    return {"labels": labels, "medoids": medoids, "cost": float(dm.min(axis=1).sum()),
            "dist_to_medoid": dm.min(axis=1), "swaps": swaps, "history": history}
