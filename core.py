"""FFR, PER, FFR-fixed and the closed-form penalty threshold for any score-based method."""
import numpy as np

TIE = 1e-12
TP = list(range(10, 91, 5))
WEIGHTS = [0.3, 0.5, 0.7]
LAMBDAS = [0.5, 1, 2, 5, 10, 20, 50, 100, 1000]


def first_argmax(x):
    i = int(np.argmax(x)); return i, bool(np.sum(x >= x[i] - TIE) > 1)


def pareto_mask(Ra, M):
    n = len(Ra)
    return np.array([not any(Ra[j] <= Ra[i] and M[j] >= M[i] and (Ra[j] < Ra[i] or M[j] > M[i]) for j in range(n)) for i in range(n)])


def lamstar(sA, v, fix, form):
    if form == 'mult':
        if sA[fix] <= 0:   # a non-positive best feasible score: any infeasible positive score beats it at every finite lambda
            return np.inf if np.any((v > 0) & (sA > sA[fix])) else 0.0
        J = np.where((v > 0) & (sA > sA[fix]))[0]
        return max([np.log(sA[j] / sA[fix]) / v[j] for j in J], default=0.0)
    J = np.where((v > 0) & (sA > sA[fix]))[0]
    return max([(sA[j] - sA[fix]) / v[j] for j in J], default=0.0)


def per_winner(sA, v, lam, form):
    """PER winner. The multiplicative form is compared in log space, log s - lambda*v, so that exp(-lambda*v)
    cannot underflow to 0 and create artificial ties; zero scores map to -inf (ties among them go to the first)."""
    if form == 'add':
        return first_argmax(sA - lam * v)
    key = np.where(sA > 0, np.log(np.where(sA > 0, sA, 1.0)) - lam * v, -np.inf)
    if np.isneginf(key).all():
        return 0, len(key) > 1
    i = int(np.argmax(key)); return i, bool(np.sum(key >= key[i] - TIE) > 1)


def analyse(X, T, w, method, forms, lams=LAMBDAS):
    """One (candidate set, limit, weights, method). Returns the lambda-independent part and per-lambda outcomes."""
    Ra = X[:, 0]; v = np.maximum(0, Ra - T) / T; F = np.where(v == 0)[0]
    assert len(F) > 0
    w = np.asarray(w)
    sA = method(X, w); sF = method(X[F], w)
    k, t1 = first_argmax(sF); ffr = int(F[k])
    k, t2 = first_argmax(sA[F]); fix = int(F[k])
    base = dict(nfeas=len(F), ffr=ffr, fix=fix, reference=ffr != fix, tie_filter=t1 or t2)
    out = {}
    for form in forms:
        ls = lamstar(sA, v, fix, form); rows = []
        for lam in lams:
            per, t3 = per_winner(sA, v, lam, form)
            rows.append(dict(lam=lam, per=per, exclusion=per != fix, tie_per=t3, viol=v[per],
                             outcome='Same' if per == ffr else ('Both' if (per != fix and ffr != fix) else ('Exclusion only' if per != fix else 'Reference only')),
                             pred_ok=(per != fix) == (lam < ls)))
        out[form] = (ls, rows)
    return base, out
