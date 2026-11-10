"""Score-based MCDM methods. Each returns scores for the rows of a candidate set, higher = better.
Criteria: column 0 = cost (Ra), column 1 = benefit (MRR). Normalisation and references come from the set passed in.
Degenerate sets (one row, or a criterion with zero range) follow the conventions documented per method."""
import numpy as np

COST = np.array([True, False])


def topsis(X, w):
    """TOPSIS, vector normalisation. Closeness = 1 when d+ + d- = 0."""
    V = X / np.linalg.norm(X, axis=0) * w
    best = np.where(COST, V.min(0), V.max(0)); worst = np.where(COST, V.max(0), V.min(0))
    dp = np.linalg.norm(V - best, axis=1); dm = np.linalg.norm(V - worst, axis=1); s = dp + dm
    return np.where(s > 0, dm / np.where(s > 0, s, 1), 1.0)


def saw(X, w):
    """Simple additive weighting, min-max normalisation (zero range -> normalised value 1). Score in [0, 1]."""
    lo, hi = X.min(0), X.max(0); rng = hi - lo
    R = np.where(rng > 0, (X - lo) / np.where(rng > 0, rng, 1), np.nan)
    R = np.where(COST, 1 - R, R); R = np.where(np.isnan(R), 1.0, R)
    return R @ w


def vikor(X, w, v=0.5):
    """VIKOR ranking index Q (compromise conditions not applied); score = 1 - Q in [0, 1].
    Zero-range criteria and zero denominators contribute 0."""
    best = np.where(COST, X.min(0), X.max(0)); worst = np.where(COST, X.max(0), X.min(0)); rng = np.abs(best - worst)
    D = np.where(rng > 0, np.abs(best - X) / np.where(rng > 0, rng, 1), 0.0) * w
    S, R = D.sum(1), D.max(1)
    def nz(a): r = a.max() - a.min(); return (a - a.min()) / r if r > 0 else np.zeros_like(a)
    return 1 - (v * nz(S) + (1 - v) * nz(R))


def promethee(X, w):
    """PROMETHEE II, usual criterion (type I) for both criteria; score = (net flow + 1) / 2 in [0, 1]; one row -> 0.5."""
    n = len(X)
    if n == 1: return np.array([0.5])
    D = X[:, None, :] - X[None, :, :]                       # a - b
    D = np.where(COST, -D, D)                               # positive = a better than b
    P = (D > 0).astype(float) @ w                           # pi(a, b)
    phi = (P.sum(1) - P.sum(0)) / (n - 1)
    return (phi + 1) / 2


def moora(X, w):
    """MOORA ratio system, vector normalisation; score can be negative (additive penalty only)."""
    N = X / np.linalg.norm(X, axis=0) * w
    return N[:, 1] - N[:, 0]


METHODS = {'TOPSIS': topsis, 'VIKOR': vikor, 'PROMETHEE II': promethee, 'SAW': saw, 'MOORA': moora}
POSITIVE = {'TOPSIS', 'VIKOR', 'PROMETHEE II', 'SAW'}      # scores in [0, 1]: multiplicative penalty admissible
