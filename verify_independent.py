"""Independent cross-check of the empirical and synthetic results.

The five methods, the Pareto filter and the FFR / FFR-fixed / PER winners are re-implemented here in plain Python
loops, without importing methods.py or core.py, directly from the definitions in Appendix B and Section 2.
PER winners are found by brute force at each lambda (multiplicative form compared in log space), not through lambda*.
The script compares every empirical case with out/ext_cases.csv and a random sample of synthetic configurations
with out/synthetic.csv.gz, and checks the case counts quoted in the manuscript.
Only the input tables are shared with the main analysis (data_sources.py).
"""
import math, random, sys
import numpy as np, pandas as pd
from data_sources import DATASETS

TIE = 1e-12
LIMITS = list(range(10, 91, 5)); WEIGHTS = [0.3, 0.5, 0.7]; LAMS = [0.5, 1, 2, 5, 10, 20, 50, 100, 1000]


# ---- methods: rows are (Ra, MRR); Ra is a cost, MRR a benefit ----
def topsis(rows, w):
    n0 = math.sqrt(sum(r[0] ** 2 for r in rows)); n1 = math.sqrt(sum(r[1] ** 2 for r in rows))
    V = [(w[0] * r[0] / n0, w[1] * r[1] / n1) for r in rows]
    ib = (min(v[0] for v in V), max(v[1] for v in V)); iw = (max(v[0] for v in V), min(v[1] for v in V))
    out = []
    for v in V:
        dp = math.hypot(v[0] - ib[0], v[1] - ib[1]); dm = math.hypot(v[0] - iw[0], v[1] - iw[1])
        out.append(dm / (dp + dm) if dp + dm > 0 else 1.0)
    return out


def saw(rows, w):
    lo = [min(r[k] for r in rows) for k in (0, 1)]; hi = [max(r[k] for r in rows) for k in (0, 1)]
    out = []
    for r in rows:
        u0 = (hi[0] - r[0]) / (hi[0] - lo[0]) if hi[0] > lo[0] else 1.0
        u1 = (r[1] - lo[1]) / (hi[1] - lo[1]) if hi[1] > lo[1] else 1.0
        out.append(w[0] * u0 + w[1] * u1)
    return out


def vikor(rows, w, v=0.5):
    best = (min(r[0] for r in rows), max(r[1] for r in rows)); worst = (max(r[0] for r in rows), min(r[1] for r in rows))
    S, R = [], []
    for r in rows:
        d = [w[k] * abs(best[k] - r[k]) / abs(best[k] - worst[k]) if best[k] != worst[k] else 0.0 for k in (0, 1)]
        S.append(sum(d)); R.append(max(d))
    def nz(a):
        lo, hi = min(a), max(a); return [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in a]
    nS, nR = nz(S), nz(R)
    return [1 - (v * a + (1 - v) * b) for a, b in zip(nS, nR)]


def promethee(rows, w):
    n = len(rows)
    if n == 1: return [0.5]
    def pi(a, b): return w[0] * (rows[a][0] < rows[b][0]) + w[1] * (rows[a][1] > rows[b][1])
    return [((sum(pi(a, b) for b in range(n) if b != a) - sum(pi(b, a) for b in range(n) if b != a)) / (n - 1) + 1) / 2 for a in range(n)]


def moora(rows, w):
    n0 = math.sqrt(sum(r[0] ** 2 for r in rows)); n1 = math.sqrt(sum(r[1] ** 2 for r in rows))
    return [w[1] * r[1] / n1 - w[0] * r[0] / n0 for r in rows]


METHODS = {'TOPSIS': topsis, 'VIKOR': vikor, 'PROMETHEE II': promethee, 'SAW': saw, 'MOORA': moora}
FORMS = {'TOPSIS': ['mult', 'add'], 'VIKOR': ['mult', 'add'], 'PROMETHEE II': ['mult', 'add'], 'SAW': ['mult', 'add'], 'MOORA': ['add']}


def first_best(vals):
    """index of the first maximum, treating values within TIE as equal"""
    m = max(vals); return next(i for i, x in enumerate(vals) if x >= m - TIE)


def pareto(rows):
    return [i for i, a in enumerate(rows) if not any((b[0] <= a[0] and b[1] >= a[1]) and (b[0] < a[0] or b[1] > a[1]) for b in rows)]


def percentile(xs, p):
    xs = sorted(xs); h = (len(xs) - 1) * p / 100; lo = math.floor(h)
    return xs[lo] + (h - lo) * (xs[min(lo + 1, len(xs) - 1)] - xs[lo])


def decide(rows, T, w, mname, form, lams):
    v = [max(0.0, r[0] - T) / T for r in rows]
    F = [i for i in range(len(rows)) if v[i] == 0]
    sA = METHODS[mname](rows, w); sF = METHODS[mname]([rows[i] for i in F], w)
    ffr = F[first_best(sF)]; fix = F[first_best([sA[i] for i in F])]
    out = []
    for lam in lams:
        if form == 'add':
            per = first_best([s - lam * vi for s, vi in zip(sA, v)])
        else:
            keys = [math.log(s) - lam * vi if s > 0 else -math.inf for s, vi in zip(sA, v)]
            per = first_best(keys) if max(keys) > -math.inf else 0
        out.append((lam, ffr, fix, per))
    return out


def outcome(ffr, fix, per):
    if per == ffr: return 'Same'
    if per != fix and ffr != fix: return 'Both'
    return 'Exclusion only' if per != fix else 'Reference only'


def empirical():
    rows_out = []
    for key, f in DATASETS.items():
        d = f(); X = list(zip(d.Ra.astype(float), d.MRR.astype(float)))
        sets = {'All runs': list(range(len(X))), 'Pareto only': pareto(X)}
        for sname, cand in sets.items():
            rows = [X[i] for i in cand]
            for tp in LIMITS:
                T = percentile([x[0] for x in X], tp)
                for wr in WEIGHTS:
                    for m in METHODS:
                        for form in FORMS[m]:
                            for lam, ffr, fix, per in decide(rows, T, (wr, 1 - wr), m, form, LAMS):
                                rows_out.append(dict(ds=key, set=sname, tp=tp, w=wr, method=m, form=form, lam=lam,
                                                     ffr_i=cand[ffr], per_i=cand[per], outcome_i=outcome(ffr, fix, per)))
    return pd.DataFrame(rows_out)


def synthetic_sample(k=3000, seed=11):
    """Regenerate the synthetic candidate sets exactly as in extended.synthetic (same generator calls and seed)
    and check k random configurations by brute force at lambda = 1, 10, 100."""
    S = pd.read_csv('out/synthetic.csv.gz')
    rng = np.random.default_rng(7); sets = {}
    for n in (9, 18, 27, 54, 108):
        for rho in (0.0, 0.5, 0.9):
            C = np.array([[0.16, rho * 0.24], [rho * 0.24, 0.36]])
            for r in range(200):
                sets[(n, rho, r)] = np.exp(rng.multivariate_normal([0, 0], C, size=n))
    pick = S.sample(k, random_state=seed); bad = 0
    for _, c in pick.iterrows():
        X = [tuple(x) for x in sets[(c.n, c.rho, c.rep)]]
        cand = list(range(len(X))) if c.set == 'All' else pareto(X)
        rows = [X[i] for i in cand]; T = percentile([x[0] for x in X], c.tp)
        res = decide(rows, T, (c.w, 1 - c.w), c.method, c.form, [1, 10, 100])
        for lam, ffr, fix, per in res:
            if (outcome(ffr, fix, per) != 'Same') != bool(c[f'differ_l{lam}']): bad += 1
        if (res[0][1] != res[0][2]) != bool(c.reference): bad += 1
    return k, bad


if __name__ == '__main__':
    ok = True
    mine = empirical(); ref = pd.read_csv('out/ext_cases.csv')
    key = ['ds', 'set', 'tp', 'w', 'method', 'form', 'lam']
    m = ref.merge(mine, on=key, how='outer', indicator=True)
    same = (m._merge == 'both').all() and (m.ffr == m.ffr_i).all() and (m.per == m.per_i).all() and (m.outcome == m.outcome_i).all()
    print(f'empirical: {len(mine)} cases recomputed; all winners and outcomes identical to ext_cases.csv: {same}')
    ok &= same and len(mine) == 41310
    a = mine[(mine.set == 'All runs') & (mine.method == 'TOPSIS') & (mine.form == 'mult')]
    print('TOPSIS x all runs: same', (a.outcome_i == 'Same').sum(), 'of', len(a), '| exclusion only', (a.outcome_i == 'Exclusion only').sum(),
          '| reference only', (a.outcome_i == 'Reference only').sum())
    ok &= (a.outcome_i == 'Same').sum() == 2064 and len(a) == 2295
    dis = mine[(mine.set == 'All runs') & (mine.outcome_i != 'Same')]
    print('all-runs disagreements', len(dis), '| Pareto-only', (mine[(mine.set == 'Pareto only') & (mine.outcome_i != 'Same')]).shape[0])
    ok &= len(dis) == 4830
    k, bad = synthetic_sample()
    print(f'synthetic: {k} random configurations re-solved by brute force at lambda = 1, 10, 100; mismatches: {bad}')
    ok &= bad == 0
    S = pd.read_csv('out/synthetic.csv.gz', usecols=['n']); print('synthetic configurations', len(S), '| x 3 lambdas =', 3 * len(S))
    ok &= len(S) == 2754000
    print('ALL CHECKS PASSED' if ok else 'CHECK FAILED'); sys.exit(0 if ok else 1)
