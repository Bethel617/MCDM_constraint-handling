"""Main script: reproduces every table, figure and number in the manuscript.

    python run_all.py            -> writes all outputs to ./out/

Conventions (stated in Section 2 of the manuscript)
  * Candidate order = source-table order; ties in any argmax go to the first candidate in that order.
  * Feasibility: Ra <= T (inclusive, no tolerance).
  * Limits T: percentiles 10, 15, ..., 90 of the dataset's observed Ra, linear interpolation
    (numpy.percentile, method='linear').
  * TOPSIS: vector normalisation, weighted; if d+ + d- = 0 (one candidate, or all candidates identical)
    the closeness coefficient is defined as 1 for every candidate.
  * PER score: C_i(A) * exp(-lambda * v_i), v_i = max(0, Ra_i - T) / T.
"""
import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from data_sources import DATASETS, SOURCES, validate, dataset_E
from core import per_winner

TP = list(range(10, 91, 5))
WEIGHTS = [0.3, 0.5, 0.7]
LAMBDAS = [0.5, 1, 2, 5, 10, 20, 50, 100, 1000]
TIE = 1e-12
OUT = 'out/'


def topsis(Ra, M, w):
    """closeness coefficients of all rows of (Ra, M); Ra cost, MRR benefit"""
    X = np.column_stack([Ra, M]).astype(float)
    V = X / np.linalg.norm(X, axis=0) * np.asarray(w)
    best = np.array([V[:, 0].min(), V[:, 1].max()]); worst = np.array([V[:, 0].max(), V[:, 1].min()])
    dp = np.linalg.norm(V - best, axis=1); dm = np.linalg.norm(V - worst, axis=1)
    s = dp + dm
    return np.where(s > 0, dm / np.where(s > 0, s, 1), 1.0)


def pareto_mask(Ra, M):
    n = len(Ra)
    return np.array([not any(Ra[j] <= Ra[i] and M[j] >= M[i] and (Ra[j] < Ra[i] or M[j] > M[i]) for j in range(n)) for i in range(n)])


def first_argmax(x):
    i = int(np.argmax(x)); tied = int(np.sum(x >= x[i] - TIE)) > 1
    return i, tied


def decide(Ra, M, T, w, lams):
    """returns per-lambda records for one (candidate set, T, w)"""
    n = len(Ra); v = np.maximum(0, Ra - T) / T; F = np.where(v == 0)[0]
    assert len(F) > 0, 'empty feasible set'
    CA = topsis(Ra, M, w)
    CF = topsis(Ra[F], M[F], w)
    k, tie_ffr = first_argmax(CF); ffr = F[k]
    k, tie_fix = first_argmax(CA[F]); fix = F[k]
    assert CA[fix] > 0, 'best feasible full-set score must be positive'
    inf = np.where((v > 0) & (CA > CA[fix]))[0]
    lamstar = max([np.log(CA[j] / CA[fix]) / v[j] for j in inf], default=0.0)
    recs = []
    for lam in lams:
        per, tie_per = per_winner(CA, v, lam, "mult")
        excl, ref = per != fix, fix != ffr
        outcome = 'Same' if per == ffr else ('Both' if excl and ref else ('Exclusion only' if excl else 'Reference only'))
        recs.append(dict(lam=lam, nfeas=len(F), single_feasible=len(F) == 1, ffr=ffr, fix=fix, per=per, outcome=outcome,
                         exclusion=excl, reference=ref, lamstar=lamstar, pred_exclusion=lam < lamstar,
                         lam_margin=abs(lam - lamstar) if lamstar > 0 else np.inf, ties=tie_ffr or tie_fix or tie_per,
                         viol_rel=v[per]))
    return recs


def run_grid(datasets):
    rows = []
    for key, d in datasets.items():
        Ra0, M0 = d.Ra.values.astype(float), d.MRR.values.astype(float)
        sets = {'All runs': np.arange(len(d)), 'Pareto only': np.where(pareto_mask(Ra0, M0))[0]}
        for sname, cand in sets.items():
            Ra, M = Ra0[cand], M0[cand]
            for tp in TP:
                T = float(np.percentile(Ra0, tp, method='linear'))
                for w in WEIGHTS:
                    for r in decide(Ra, M, T, (w, 1 - w), LAMBDAS):
                        for c in ('ffr', 'fix', 'per'): r[c] = int(cand[r[c]])
                        rows.append(dict(ds=key, set=sname, tp=tp, T=T, w_Ra=w, **r))
    return pd.DataFrame(rows)


def consequences(R, datasets):
    out = []
    for _, r in R[(R.set == 'All runs') & (R.outcome != 'Same')].iterrows():
        d = datasets[r.ds]; a, b = d.iloc[r.ffr], d.iloc[r.per]
        out.append(dict(ds=r.ds, tp=r.tp, w_Ra=r.w_Ra, lam=r.lam, outcome=r.outcome,
                        dRa=b.Ra - a.Ra, dMRR=b.MRR - a.MRR,
                        dRa_pct=100 * (b.Ra - a.Ra) / np.ptp(d.Ra), dMRR_pct=100 * (b.MRR - a.MRR) / np.ptp(d.MRR),
                        exceed_pct=100 * r.viol_rel, d_speed=b.speed - a.speed, d_f=b.f - a.f, d_ap=b.ap - a.ap,
                        category_change=f'{a.category}->{b.category}' if a.category != b.category else '',
                        ffr_condition=f'{a.speed}/{a.f}/{a.ap} {a.category}'.strip(),
                        per_condition=f'{b.speed}/{b.f}/{b.ap} {b.category}'.strip()))
    return pd.DataFrame(out)


def e_resolution(n_draws=500, seed=1):
    """dataset E: each mass loss is the difference of two readings, each rounded to 1 g. The true masses lie within
    +/-0.5 g of each reading, so the error of the difference is e_before - e_after with e ~ U(-0.5, 0.5): triangular on (-1, 1) g.
    Mass losses are floored at 0.05 g. The primary analysis (TOPSIS, multiplicative, all runs) is repeated for each draw."""
    rng = np.random.default_rng(seed); res = []
    base = dataset_E()
    for k in range(n_draws):
        noise = rng.uniform(-0.5, 0.5, 27) - rng.uniform(-0.5, 0.5, 27)
        d = dataset_E(dm_override=lambda m: np.maximum(m.dm + noise[:len(m)], 0.05))
        R = run_grid({'E': d}); R = R[R.set == 'All runs']
        res.append(dict(draw=k, same=(R.outcome == 'Same').mean(), exclusion=R.exclusion.mean(), reference=R.reference.mean(),
                        pareto=','.join(map(str, d.group[pareto_mask(d.Ra.values, d.MRR.values)]))))
    return pd.DataFrame(res)


def pct(x): return round(100 * x, 1)


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    datasets = {k: f() for k, f in DATASETS.items()}
    for k, d in datasets.items(): validate(k, d)
    pd.concat([d.assign(dataset=k, source=SOURCES[k]) for k, d in datasets.items()]).to_csv(OUT + 'consolidated_datasets.csv', index=False)

    R = run_grid(datasets); R.to_csv(OUT + 'cases.csv', index=False)
    S = {}
    # checks of the analytical results
    S['threshold_matches_all_cases'] = bool((R.pred_exclusion == R.exclusion).all())
    S['min_grid_distance_to_lamstar'] = float(R.lam_margin.min())
    S['per_equals_fix_at_lam1000'] = bool((R[R.lam == 1000].per == R[R.lam == 1000].fix).all())
    S['cases_with_ties'] = int(R.ties.sum()); S['single_feasible_cases'] = int(R.single_feasible.sum())
    S['empty_feasible_sets'] = 0
    # outcome counts
    tab = R.groupby(['set', 'ds']).outcome.value_counts().unstack(fill_value=0)
    tab.to_csv(OUT + 'table_outcomes_counts.csv')
    tot = R.groupby('set').outcome.value_counts().unstack(fill_value=0); tot.to_csv(OUT + 'table_outcomes_totals.csv')
    S['totals'] = {s: {o: int(tot.loc[s].get(o, 0)) for o in ['Same', 'Exclusion only', 'Reference only', 'Both']} for s in tot.index}
    S['any_reference'] = {s: int(R[(R.set == s) & R.reference & (R.outcome != 'Same')].shape[0]) for s in tot.index}
    S['any_exclusion'] = {s: int(R[(R.set == s) & R.exclusion].shape[0]) for s in tot.index}
    S['n_cases_per_set'] = int((R.set == 'All runs').sum())
    A = R[R.set == 'All runs']
    S['disagree_by_lambda'] = {str(l): pct(v) for l, v in (A.outcome != 'Same').groupby(A.lam).mean().items()}
    tb = pd.cut(A.tp, [0, 25, 50, 75, 95], labels=['P10-25', 'P30-50', 'P55-75', 'P80-90'])
    fb = pd.cut(A.nfeas, [0, 3, 6, 10, 30], labels=['1-3', '4-6', '7-10', '>10'])
    S['disagree_by_tightness'] = {str(k): pct(v) for k, v in (A.outcome != 'Same').groupby(tb).mean().items()}
    S['disagree_by_nfeas'] = {str(k): pct(v) for k, v in (A.outcome != 'Same').groupby(fb).mean().items()}
    L = R.drop_duplicates(['ds', 'set', 'tp', 'w_Ra'])
    S['lamstar'] = {s: {k: dict(share_positive=pct((g.lamstar > 0).mean()), median_positive=round(float(g[g.lamstar > 0].lamstar.median()), 2) if (g.lamstar > 0).any() else 0,
                               max=round(float(g.lamstar.max()), 2)) for k, g in L[L.set == s].groupby('ds')} for s in ['All runs', 'Pareto only']}
    # consequences
    Cq = consequences(R, datasets); Cq.to_csv(OUT + 'consequences_cases.csv', index=False)
    ct = Cq.groupby(['ds', 'outcome']).agg(cases=('dRa', 'size'), dRa=('dRa', 'median'), dMRR=('dMRR', 'median'), dRa_pct=('dRa_pct', 'median'),
                                          dMRR_pct=('dMRR_pct', 'median'), exceed_med=('exceed_pct', 'median'), exceed_max=('exceed_pct', 'max'))
    ct.round(3).to_csv(OUT + 'table_consequences.csv')
    S['consequence'] = dict(all_higher_MRR_and_Ra=bool(((Cq.dRa > 0) & (Cq.dMRR > 0)).all()), feed_up=pct((Cq.d_f > 0).mean()),
                            ap_changed=pct((Cq.d_ap != 0).mean()), speed_changed=pct((Cq.d_speed != 0).mean()),
                            category_changed=pct((Cq.category_change != '').mean()),
                            category_changes=Cq[Cq.category_change != ''].groupby(['ds', 'category_change']).size().to_dict())
    S['consequence']['category_changes'] = {f'{k[0]}: {k[1]}': int(v) for k, v in S['consequence']['category_changes'].items()}
    # dataset E measurement resolution
    E = e_resolution(); E.to_csv(OUT + 'E_mass_resolution_draws.csv', index=False)
    base = A[A.ds == 'E']
    S['E_resolution'] = dict(draws=len(E), same_base=pct((base.outcome == 'Same').mean()), same_min=pct(E.same.min()), same_max=pct(E.same.max()),
                             same_p05=pct(E.same.quantile(.05)), same_p95=pct(E.same.quantile(.95)),
                             reference_draws_with_any=int((E.reference > 0).sum()),
                             pareto_sets=E.pareto.value_counts().head(5).to_dict())
    json.dump(S, open(OUT + 'summary.json', 'w'), indent=1, default=str)
    print(json.dumps(S, indent=1, default=str)); print(tab); print(ct.round(2))
    import make_figures; make_figures.main(datasets, R, Cq)
    # generality analyses: five methods x two penalty forms, and the synthetic study
    import extended, ext_figures
    Ex, Th = extended.empirical(datasets); Ex.to_csv(OUT + 'ext_cases.csv', index=False); Th.to_csv(OUT + 'ext_thresholds.csv', index=False)
    assert Ex.pred_ok.all(), 'Proposition 2 violated'
    ext_figures.fig_methods(Ex, OUT + 'Fig3_methods.png')
    if not os.path.exists(OUT + 'synthetic.csv.gz') or os.environ.get('RERUN_SYNTHETIC'):
        extended.synthetic().to_csv(OUT + 'synthetic.csv.gz', index=False)   # ~10 min
    ext_figures.fig_synthetic(pd.read_csv(OUT + 'synthetic.csv.gz'), OUT + 'Fig5_synthetic.png')


if __name__ == '__main__':
    main()
