"""Original (Hinton & Roweis) SNE with perplexity-calibrated bandwidths."""
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.metrics import pairwise_distances


def conditional_P(Z, perplexity=None, sigma=None, iters=50):
    """Row-stochastic P[j|i]. Give `perplexity` (per-point sigma by bisection) or a fixed `sigma`.

    Returns (P, effective_perplexity_per_point).
    """
    D = pairwise_distances(Z, squared=True)
    n = len(D)
    if perplexity is not None:
        if not 1 < perplexity < n - 1:
            raise ValueError("perplexity must be in (1, n-1)")
        target = np.log(perplexity)
        lo = np.full(n, -20.0)
        hi = np.full(n, 20.0)
        for _ in range(iters):
            mid = (lo + hi) / 2
            H = _entropy(D, np.exp(mid))
            too_wide = H > target
            lo = np.where(too_wide, mid, lo)
            hi = np.where(too_wide, hi, mid)
        beta = np.exp((lo + hi) / 2)
    else:
        beta = np.full(n, 1.0 / (2 * (sigma if sigma else 1.0) ** 2))
    logits = -beta[:, None] * D
    np.fill_diagonal(logits, -np.inf)
    logits -= logsumexp(logits, axis=1, keepdims=True)
    P = np.exp(logits)
    nz = P > 0
    H = -np.sum(np.where(nz, P * np.log(np.where(nz, P, 1)), 0), axis=1)
    return P, np.exp(H)


def _entropy(D, beta):
    logits = -beta[:, None] * D
    np.fill_diagonal(logits, -np.inf)
    logZ = logsumexp(logits, axis=1)
    P = np.exp(logits - logZ[:, None])
    return logZ + beta * np.sum(P * D, axis=1)


def sne_objective(y_flat, P, n, dims=2):
    Y = y_flat.reshape(n, dims)
    sq = np.sum(Y * Y, axis=1)
    d = np.maximum(sq[:, None] + sq[None, :] - 2 * (Y @ Y.T), 0)
    log_q = -d
    np.fill_diagonal(log_q, -np.inf)
    log_q -= logsumexp(log_q, axis=1, keepdims=True)
    Q = np.exp(log_q)
    nz = P > 0
    loss = np.sum(P[nz] * (np.log(P[nz]) - log_q[nz]))
    diff = P - Q
    w = diff + diff.T
    grad = 2 * (w.sum(axis=1, keepdims=True) * Y - w @ Y)
    return loss, grad.ravel()


def fit_sne(Z, perplexity=None, sigma=None, seed=42, maxiter=300):
    """Optimise original SNE with L-BFGS (same settings as the original notebook)."""
    t0 = time.time()
    n = len(Z)
    P, eff = conditional_P(Z, perplexity=perplexity, sigma=sigma)
    rng = np.random.RandomState(seed)
    y0 = rng.normal(0, 0.01, size=(n, 2))
    res = minimize(sne_objective, y0.ravel(), args=(P, n, 2), method="L-BFGS-B",
                   jac=True, options={"maxiter": maxiter, "ftol": 1e-8})
    return {"Y": res.x.reshape(n, 2), "loss": float(res.fun), "nit": int(res.nit),
            "eff_perplexity": float(np.mean(eff)), "seconds": time.time() - t0}
