"""Figures 1, 2, 4 and 6 of the manuscript (3 and 5 are in ext_figures.py), from the outputs of run_all.py."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#52514e', 'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'savefig.dpi': 300, 'savefig.bbox': 'tight'})
INK, INK2, GREY, LIGHT, BLUE, ORANGE = '#0b0b0b', '#52514e', '#9e9d98', '#efeeea', '#2a78d6', '#eb6834'
DSC = {'A': '#2a78d6', 'B': '#eb6834', 'C': '#1baf7a', 'D': '#eda100', 'E': '#6250d6'}
DSM = {'A': 'o', 'B': 's', 'C': '^', 'D': 'D', 'E': 'v'}
TITLES = {'A': 'A  (L27, n = 27)', 'B': 'B  (L18, MQL, n = 18)', 'C': 'C  (BBD, wet, n = 13)', 'D': 'D  (L9, n = 9)', 'E': 'E  (present, dry, n = 27)'}
O = 'out/'
shade = lambda base, a: tuple(1 - a * (1 - b) for b in to_rgb(base))


def main(datasets, R, Cq):
    from run_all import pareto_mask
    A = R[R.set == 'All runs']
    # Fig 1
    fig, ax = plt.subplots(1, 5, figsize=(7.2, 1.95))
    for a, k in zip(ax, datasets):
        d = datasets[k]; Ra, M = d.Ra.values, d.MRR.values / 1000; p = pareto_mask(d.Ra.values, d.MRR.values)
        a.scatter(Ra[~p], M[~p], s=12, color=GREY, lw=0, zorder=2); o = np.argsort(Ra[p])
        a.plot(Ra[p][o], M[p][o], color=BLUE, lw=1.2, zorder=3); a.scatter(Ra[p], M[p], s=18, color=BLUE, edgecolor='white', lw=0.6, zorder=4)
        a.set_title(TITLES[k], loc='left', fontsize=7.8); a.set_xlabel('Ra (µm)'); a.tick_params(length=2)
    ax[0].set_ylabel('MRR (10³ mm³/min)')
    fig.text(0.99, -0.12, 'Blue: Pareto-optimal runs (min Ra, max MRR); grey: dominated runs.', ha='right', fontsize=7, color=INK2)
    fig.tight_layout(w_pad=0.6); fig.savefig(O + 'Fig1_datasets.png'); plt.close(fig)
    # Fig 2
    LAM = sorted(A.lam.unique()); TPs = sorted(A.tp.unique())
    fig, ax = plt.subplots(1, 5, figsize=(7.2, 2.25), sharey=True)
    for a, k in zip(ax, datasets):
        s = A[A.ds == k]
        for i, tp in enumerate(TPs):
            for j, lam in enumerate(LAM):
                c = s[(s.tp == tp) & (s.lam == lam)]; nex = c.exclusion.sum(); nref = (c.reference & (c.outcome != 'Same')).sum()
                col = LIGHT if nex == 0 and nref == 0 else shade(ORANGE if nex >= nref else BLUE, {1: .35, 2: .65, 3: 1.0}[max(nex, nref)])
                a.add_patch(plt.Rectangle((i, j), 0.94, 0.94, color=col, lw=0))
        a.set_xlim(0, len(TPs)); a.set_ylim(0, len(LAM)); a.grid(False)
        a.set_xticks([0.5, 3.5, 8.5, 13.5, 16.5]); a.set_xticklabels(['10', '25', '50', '75', '90'])
        a.set_title(TITLES[k], loc='left', fontsize=7.8); a.tick_params(length=0)
        for sp in a.spines.values(): sp.set_visible(False)
    fig.supxlabel("Roughness limit T (percentile of the dataset's observed Ra)", fontsize=8.5, y=0.06)
    ax[0].set_yticks(np.arange(len(LAM)) + 0.47); ax[0].set_yticklabels([f'{l:g}' for l in LAM]); ax[0].set_ylabel('Penalty strength λ')
    h = [Patch(color=LIGHT, label='Same winner (all 3 weightings)')] + [Patch(color=shade(ORANGE, a_), label=f'Exclusion effect, {m} of 3 weightings') for m, a_ in [(1, .35), (2, .65), (3, 1)]] + \
        [Patch(color=shade(BLUE, .35), label='Reference effect, 1 of 3 weightings')]
    fig.legend(handles=h, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.16), fontsize=6.8, handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(w_pad=0.4); fig.savefig(O + 'Fig2_decision_maps.png'); plt.close(fig)
    # Fig 3
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    a = ax[0]; g = A.groupby('lam'); x = np.arange(len(LAM))
    rf = g.apply(lambda s: (s.reference & (s.outcome != 'Same')).mean() * 100).values
    ex = g.apply(lambda s: (s.exclusion & ~s.reference).mean() * 100).values
    a.bar(x, rf, 0.62, color=BLUE, label='Reference effect'); a.bar(x, ex, 0.62, bottom=rf, color=ORANGE, label='Exclusion effect')
    for xi, (r_, e_) in enumerate(zip(rf, ex)): a.text(xi, r_ + e_ + 0.5, f'{r_ + e_:.1f}', ha='center', fontsize=6.5, color=INK)
    a.set_xticks(x); a.set_xticklabels([f'{l:g}' for l in LAM], fontsize=6.8); a.set_xlabel('Penalty strength λ')
    a.set_ylabel('Cases where winners differ (%)'); a.set_title('(a) By penalty strength', loc='left', fontsize=8.3); a.legend(frameon=False, loc='upper right'); a.set_ylim(0, 40)
    dis = A.outcome != 'Same'
    d1 = dis.groupby(pd.cut(A.tp, [0, 25, 50, 75, 95], labels=['P10–25', 'P30–50', 'P55–75', 'P80–90'])).mean() * 100
    d2 = dis.groupby(pd.cut(A.nfeas, [0, 3, 6, 10, 30], labels=['1–3', '4–6', '7–10', '>10'])).mean() * 100
    for a, d, ttl, xl in [(ax[1], d1, '(b) By limit tightness', 'Limit T (Ra percentile)'), (ax[2], d2, '(c) By feasible runs', 'Number of feasible runs')]:
        a.bar(np.arange(4), d.values, 0.6, color=[BLUE if v == d.max() else GREY for v in d.values])
        for xi, v in enumerate(d.values): a.text(xi, v + 0.6, f'{v:.0f}', ha='center', fontsize=6.8, color=INK)
        a.set_xticks(np.arange(4)); a.set_xticklabels(d.index, fontsize=6.8); a.set_xlabel(xl); a.set_ylim(0, 40)
        a.set_title(ttl, loc='left', fontsize=8.3); a.tick_params(labelleft=False)
    fig.tight_layout(w_pad=0.8); fig.savefig(O + 'Fig4_drivers.png'); plt.close(fig)
    # Fig 4
    u = Cq.drop_duplicates(['ds', 'tp', 'w_Ra', 'dRa', 'dMRR'])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1]))
    a = ax[0]
    for k in datasets:
        s = u[u.ds == k]
        if len(s): a.scatter(s.dRa_pct, s.dMRR_pct, s=26, marker=DSM[k], facecolor='none', edgecolor=DSC[k], lw=1.1, label=k)
    a.axhline(0, color=INK2, lw=0.6); a.axvline(0, color=INK2, lw=0.6)
    a.set_xlabel('ΔRa, PER − FFR (% of dataset range)'); a.set_ylabel('ΔMRR, PER − FFR (% of range)')
    a.set_title('(a) Every PER deviation buys MRR with roughness', loc='left', fontsize=8.3)
    a.legend(frameon=False, title='Dataset', title_fontsize=7, loc='upper right', bbox_to_anchor=(1, 0.93), handletextpad=0.2, ncol=2); a.set_xlim(left=0); a.set_ylim(bottom=0)
    a.text(0.98, 0.98, 'Coincident cases overplot', transform=a.transAxes, ha='right', va='top', fontsize=6.5, color=INK2)
    b = ax[1]; ex = Cq[Cq.outcome.str.contains('Exclusion')]; ks = [k for k in datasets if (ex.ds == k).any()]
    bp = b.boxplot([ex[ex.ds == k].exceed_pct.values for k in ks], widths=0.5, patch_artist=True, medianprops=dict(color=INK, lw=1.2),
                   whiskerprops=dict(color=INK2), capprops=dict(color=INK2), flierprops=dict(marker='o', ms=3, mec=INK2))
    for p_, k in zip(bp['boxes'], ks): p_.set(facecolor=DSC[k], alpha=0.35, edgecolor=INK2)
    b.set_xticks(range(1, len(ks) + 1)); b.set_xticklabels(ks); b.set_xlabel('Dataset (exclusion cases only)')
    b.set_ylabel('Limit exceedance of PER winner (% of T)'); b.set_title('(b) Size of the accepted violation', loc='left', fontsize=8.3)
    fig.tight_layout(); fig.savefig(O + 'Fig6_consequence.png'); plt.close(fig)
