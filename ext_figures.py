"""Figures for the generality analyses (method comparison, synthetic study)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from make_figures import INK, INK2, GREY, LIGHT, BLUE, ORANGE
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 8.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#52514e', 'axes.grid': True, 'grid.color': '#e4e3df',
                     'grid.linewidth': 0.6, 'axes.axisbelow': True, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
PURPLE = '#6250d6'
ORDER = [('TOPSIS', 'mult'), ('TOPSIS', 'add'), ('MOORA', 'add'), ('VIKOR', 'mult'), ('VIKOR', 'add'),
         ('SAW', 'mult'), ('SAW', 'add'), ('PROMETHEE II', 'mult'), ('PROMETHEE II', 'add')]
LBL = {'mult': '×', 'add': '+'}
METH = ['TOPSIS', 'MOORA', 'VIKOR', 'SAW', 'PROMETHEE II']


def fig_methods(E, path):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for a, s, ttl in zip(ax, ['All runs', 'Pareto only'], ['(a) All measured runs as candidates', '(b) Pareto-optimal runs only']):
        d = E[E.set == s]; y = np.arange(len(ORDER))[::-1]
        for yi, (m, f) in zip(y, ORDER):
            g = d[(d.method == m) & (d.form == f)]; n = len(g)
            ex = (g.outcome == 'Exclusion only').sum() / n * 100; rf = (g.outcome == 'Reference only').sum() / n * 100; bo = (g.outcome == 'Both').sum() / n * 100
            a.barh(yi, rf, color=BLUE, height=0.62); a.barh(yi, bo, left=rf, color=PURPLE, height=0.62); a.barh(yi, ex, left=rf + bo, color=ORANGE, height=0.62)
            a.text(rf + bo + ex + 0.8, yi, f'{rf + bo + ex:.1f}', va='center', fontsize=7, color=INK)
        a.set_yticks(y); a.set_yticklabels([f'{m} ({LBL[f]})' for m, f in ORDER], fontsize=7.5)
        a.set_xlim(0, 45); a.set_xlabel('Cases where FFR and PER differ (%)'); a.set_title(ttl, loc='left', fontsize=8.3)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=BLUE, label='Reference effect only'), Patch(color=PURPLE, label='Both effects'), Patch(color=ORANGE, label='Exclusion effect only')],
               loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.55, -0.06), fontsize=7.5)
    fig.text(0.01, -0.06, '× multiplicative, + additive penalty', fontsize=7, color=INK2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_synthetic(S, path):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=False)
    cols = {'TOPSIS': BLUE, 'MOORA': '#1baf7a', 'VIKOR': ORANGE, 'SAW': '#eda100', 'PROMETHEE II': PURPLE}
    mk = {'TOPSIS': 'o', 'MOORA': 's', 'VIKOR': '^', 'SAW': 'D', 'PROMETHEE II': 'v'}
    base = S[S.form == 'add'].drop_duplicates(['n', 'rho', 'rep', 'set', 'tp', 'w', 'method'])
    for a, s, ttl in zip(ax[:2], ['All', 'Pareto'], ['(a) Reference effect, all alternatives', '(b) Reference effect, Pareto subset']):
        for m in METH:
            g = base[(base.set == s) & (base.method == m)].groupby('n').reference.mean() * 100
            a.plot(g.index, g.values, marker=mk[m], color=cols[m], lw=1.4, ms=4, label=m)
        a.set_xscale('log'); a.set_xticks([9, 18, 27, 54, 108]); a.set_xticklabels([9, 18, 27, 54, 108]); a.set_xlabel('Candidate-set size n')
        a.set_title(ttl, loc='left', fontsize=8.1); a.set_ylim(0, None)
    ax[0].set_ylabel('Cases (%)'); ax[0].legend(frameon=False, fontsize=6.5)
    a = ax[2]
    t = S[(S.set == 'All') & (S.method == 'TOPSIS') & (S.form == 'mult')]
    shades = {0.0: 0.35, 0.5: 0.65, 0.9: 1.0}
    for rho in (0.0, 0.5, 0.9):
        g = t[t.rho == rho].groupby('n').differ_l10.mean() * 100
        a.plot(g.index, g.values, marker='o', ms=4, lw=1.4, color=BLUE, alpha=shades[rho], label=f'ρ = {rho:g}')
    a.set_xscale('log'); a.set_xticks([9, 18, 27, 54, 108]); a.set_xticklabels([9, 18, 27, 54, 108]); a.set_xlabel('Candidate-set size n')
    a.set_title('(c) Any disagreement, TOPSIS (×), λ = 10', loc='left', fontsize=8.1); a.legend(frameon=False, fontsize=6.5); a.set_ylim(0, None)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == '__main__':
    E = pd.read_csv('out/ext_cases.csv'); fig_methods(E, 'out/Fig3_methods.png')
    import os
    if os.path.exists('out/synthetic.csv.gz'): fig_synthetic(pd.read_csv('out/synthetic.csv.gz'), 'out/Fig5_synthetic.png')
