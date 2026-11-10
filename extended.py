"""Generality analyses: five MCDM methods x two penalty forms on the five datasets, and a synthetic study."""
import numpy as np, pandas as pd, json, time
from methods import METHODS, POSITIVE
from core import analyse, pareto_mask, TP, WEIGHTS, LAMBDAS
from data_sources import DATASETS, validate

COMBOS = [(m, f) for m in METHODS for f in (['mult', 'add'] if m in POSITIVE else ['add'])]


def empirical(datasets):
    rows, thr = [], []
    for key, d in datasets.items():
        X0 = d[['Ra', 'MRR']].values.astype(float)
        sets = {'All runs': np.arange(len(d)), 'Pareto only': np.where(pareto_mask(X0[:, 0], X0[:, 1]))[0]}
        for sname, cand in sets.items():
            X = X0[cand]
            for tp in TP:
                T = float(np.percentile(X0[:, 0], tp, method='linear'))
                for w in WEIGHTS:
                    for mname, mf in METHODS.items():
                        forms = ['mult', 'add'] if mname in POSITIVE else ['add']
                        base, out = analyse(X, T, (w, 1 - w), mf, forms)
                        for form, (ls, rr) in out.items():
                            thr.append(dict(ds=key, set=sname, tp=tp, w=w, method=mname, form=form, lamstar=ls, reference=base['reference'], tie=base['tie_filter']))
                            for r in rr:
                                rows.append(dict(ds=key, set=sname, tp=tp, w=w, method=mname, form=form, nfeas=base['nfeas'],
                                                 ffr=int(cand[base['ffr']]), per=int(cand[r['per']]), lam=r['lam'], outcome=r['outcome'],
                                                 exclusion=r['exclusion'], reference=base['reference'], pred_ok=r['pred_ok'],
                                                 tie=base['tie_filter'] or r['tie_per'], viol=r['viol']))
    return pd.DataFrame(rows), pd.DataFrame(thr)


def synthetic(ns=(9, 18, 27, 54, 108), rhos=(0.0, 0.5, 0.9), reps=200, seed=7):
    """Random candidate sets: log Ra and log MRR bivariate normal (sd 0.4 and 0.6), correlation rho (rho > 0 = conflicting
    objectives: productive conditions are rougher). Outcomes for the lambda grid follow from Proposition 2."""
    rng = np.random.default_rng(seed); rows = []
    for n in ns:
        for rho in rhos:
            C = np.array([[0.16, rho * 0.24], [rho * 0.24, 0.36]])
            for r in range(reps):
                X = np.exp(rng.multivariate_normal([0, 0], C, size=n))
                pm = pareto_mask(X[:, 0], X[:, 1])
                for sname, cand in (('All', np.arange(n)), ('Pareto', np.where(pm)[0])):
                    Xs = X[cand]
                    for tp in TP:
                        T = float(np.percentile(X[:, 0], tp, method='linear'))
                        if not (Xs[:, 0] <= T).any(): continue
                        for w in WEIGHTS:
                            for mname, mf in METHODS.items():
                                forms = ['mult', 'add'] if mname in POSITIVE else ['add']
                                base, out = analyse(Xs, T, (w, 1 - w), mf, forms, lams=[])
                                for form, (ls, _) in out.items():
                                    rows.append(dict(n=n, rho=rho, rep=r, set=sname, n_pareto=int(pm.sum()), tp=tp, w=w, method=mname, form=form,
                                                     nfeas=base['nfeas'], reference=base['reference'], lamstar=ls, tie=base['tie_filter']))
    S = pd.DataFrame(rows)
    for lam in (1, 10, 100):
        S[f'differ_l{lam}'] = S.reference | (S.lamstar > lam)
    return S


if __name__ == '__main__':
    t = time.time()
    ds = {k: f() for k, f in DATASETS.items()}
    for k, d in ds.items(): validate(k, d)
    E, Th = empirical(ds); E.to_csv('out/ext_cases.csv', index=False); Th.to_csv('out/ext_thresholds.csv', index=False)
    print('empirical', len(E), 'cases; threshold rule holds:', E.pred_ok.all(), '%.0fs' % (time.time() - t))
    S = synthetic(); S.to_csv('out/synthetic.csv.gz', index=False); print('synthetic', len(S), '%.0fs' % (time.time() - t))
