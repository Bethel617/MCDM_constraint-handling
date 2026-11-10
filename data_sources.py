"""Input data for the five Ti-6Al-4V turning datasets.

Every value below is transcribed from the cited source table (A-D) or from the authors'
original records (E). Each dataset keeps its complete condition identity, including
categorical factors (tool grade in B, cooling environment in D).

Units
  speed : m/min (A: spindle speed in rpm, as printed)
  f     : mm/rev
  ap    : mm
  Ra    : micrometre
  MRR   : mm^3/min
Nominal MRR (B, C, D):  MRR [mm^3/min] = 1000 * Vc [m/min] * f [mm/rev] * ap [mm]
"""
import numpy as np, pandas as pd

RHO_TI64 = 4.43        # g/cm^3, dataset E
CUT_LENGTH_E = 50.0    # mm, length of cut per run in dataset E (reproduces the recorded times)


def dataset_A():
    """Rajaparthiban & Naveen Sait, Mechanika 24(4) 2018, Table 3.
    MRR printed in mm^3/s (determined from weight loss); converted here x60 to mm^3/min.
    The conversion is a positive rescaling and does not affect any TOPSIS ranking."""
    S = [1000] * 9 + [1500] * 9 + [2000] * 9
    F = [.10, .10, .10, .15, .15, .15, .20, .20, .20] * 3
    D = [.50, .75, 1.00] * 9
    Ra = [2.83, 3.20, 3.64, 3.45, 4.30, 4.52, 4.25, 4.82, 5.72, 2.43, 2.93, 3.35, 3.05, 3.42, 4.05, 3.73, 4.05, 4.72,
          1.96, 2.24, 2.95, 2.82, 3.08, 3.65, 3.24, 3.82, 4.02]
    M_s = [424.34, 452.65, 472.91, 465.75, 483.96, 496.52, 485.91, 502.74, 523.86, 512.75, 534.85, 552.97, 524.85,
           542.86, 560.32, 553.64, 571.95, 592.65, 581.55, 598.53, 620.74, 596.52, 621.95, 640.75, 620.84, 642.86, 660.25]
    d = pd.DataFrame(dict(source_row=range(1, 28), speed=S, f=F, ap=D, Ra=Ra, MRR_source=M_s))
    d['MRR'] = d.MRR_source * 60.0
    d['category'] = ''
    d['MRR_type'] = 'measured (source: mm3/s)'
    return d


def dataset_B():
    """Belbellaa et al., Periodica Polytechnica Mech. Eng. 67(1) 2023, Table 4 (MQL).
    Runs 1-9: uncoated carbide H13A (M1); runs 10-18: PVD-coated carbide GC1125 (M2)."""
    rows = [(50, .08, .2, .589), (75, .12, .2, .869), (100, .16, .2, 1.249), (50, .08, .4, .673), (75, .12, .4, .755),
            (100, .16, .4, 1.250), (50, .12, .6, .845), (75, .16, .6, 1.277), (100, .08, .6, .765), (50, .16, .2, 1.191),
            (75, .08, .2, .468), (100, .12, .2, .662), (50, .12, .4, .652), (75, .16, .4, 1.236), (100, .08, .4, .483),
            (50, .16, .6, 1.066), (75, .08, .6, .467), (100, .12, .6, .570)]
    d = pd.DataFrame(rows, columns=['speed', 'f', 'ap', 'Ra']); d.insert(0, 'source_row', range(1, 19))
    d['category'] = ['H13A'] * 9 + ['GC1125'] * 9
    d['MRR'] = 1000 * d.speed * d.f * d.ap; d['MRR_type'] = 'nominal'
    return d


def dataset_C():
    """Niharika et al., Archive of Mechanical Engineering 63(4) 2016, Box-Behnken design (wet).
    The five centre-point replicates (150, 0.15, 0.5) carry identical values and are merged into one row."""
    rows = [(150, .10, .2, .68), (150, .15, .5, .71), (239, .15, .2, .46), (90.1, .15, .8, 1.16), (150, .20, .8, .78),
            (239, .10, .5, .42), (239, .20, .5, .49), (150, .20, .2, .80), (90.1, .10, .5, 1.11), (90.1, .20, .5, 1.19),
            (90.1, .15, .2, 1.17), (150, .10, .8, .69), (239, .15, .8, .48)]
    d = pd.DataFrame(rows, columns=['speed', 'f', 'ap', 'Ra']); d.insert(0, 'source_row', range(1, 14))
    d['category'] = ''
    d['MRR'] = 1000 * d.speed * d.f * d.ap; d['MRR_type'] = 'nominal'
    return d


def dataset_D():
    """Khan & Jaffery, Procedia CIRP 135 (2025), Tables 1-2, L9; ap = 1 mm (stated in text).
    Ra = mean of the two printed readings."""
    rows = [(75, .10, 'Dry', 1.51, 1.54), (125, .10, 'Wet', 1.18, 1.20), (175, .10, 'Cryogenic', .97, 1.01),
            (75, .15, 'Cryogenic', 1.70, 1.75), (125, .15, 'Dry', 1.77, 1.81), (175, .15, 'Wet', 1.53, 1.58),
            (75, .20, 'Wet', 2.85, 2.89), (125, .20, 'Cryogenic', 2.45, 2.51), (175, .20, 'Dry', 2.58, 2.63)]
    d = pd.DataFrame(rows, columns=['speed', 'f', 'category', 'Ra1', 'Ra2']); d.insert(0, 'source_row', range(1, 10))
    d['ap'] = 1.0; d['Ra'] = (d.Ra1 + d.Ra2) / 2
    d['MRR'] = 1000 * d.speed * d.f * d.ap; d['MRR_type'] = 'nominal'
    return d[['source_row', 'speed', 'f', 'ap', 'Ra', 'category', 'MRR', 'MRR_type']]


# ---------------- dataset E: authors' records ----------------
# Ra: Table 5 of Priyadarsini et al., Eng. Res. Express 8 (2026) 085409, by group (condition)
E_RA = [(1, 45, .10, .3, .712), (2, 45, .10, .1, .735), (3, 45, .15, .1, .762), (4, 45, .20, .1, 1.789), (5, 45, .15, .5, 1.001),
        (6, 45, .10, .5, .84), (7, 45, .20, .3, .626), (8, 45, .15, .3, 1.463), (9, 45, .20, .5, 1.681), (10, 70, .15, .1, .542),
        (11, 70, .10, .1, .542), (12, 70, .15, .3, .715), (13, 70, .10, .3, 1.654), (14, 70, .20, .3, 1.87), (15, 70, .20, .1, .526),
        (16, 70, .15, .5, .861), (17, 70, .10, .5, .683), (18, 70, .20, .5, .995), (19, 90, .10, .1, .639), (20, 90, .15, .5, 2.643),
        (21, 90, .20, .5, .789), (22, 90, .15, .3, 1.285), (23, 90, .10, .5, .683), (24, 90, .20, .1, 2.055), (25, 90, .15, .1, .841),
        (26, 90, .10, .3, .784), (27, 90, .20, .3, 1.514)]
# Mass-loss record: run, ap (mm), feed setting (in/rev), nominal speed level (m/min), spindle speed (rpm),
# mass before (g), mass after (g), recorded time (s, as displayed)
E_MASS = [(1, .1, .0038, 45, 235, 3210, 3207, 132), (4, .1, .0055, 45, 235, 3207, 3206, 91), (7, .1, .0076, 45, 235, 3206, 3202, 66),
          (10, .3, .0038, 45, 235, 3218, 3210, 132), (13, .3, .0055, 45, 235, 3151, 3138, 91), (16, .3, .0076, 45, 235, 3159, 3151, 66),
          (19, .5, .0038, 45, 235, 3183, 3159, 132), (22, .5, .0055, 45, 235, 3202, 3183, 91), (25, .5, .0076, 45, 235, 3124, 3108, 66),
          (2, .1, .0038, 70, 385, 3132, 3125, 81), (5, .1, .0055, 70, 385, 3138, 3132, 56), (8, .1, .0076, 70, 385, 3072, 3068, 40),
          (11, .3, .0038, 70, 385, 3093, 3083, 81), (14, .3, .0055, 70, 385, 3108, 3093, 56), (17, .3, .0076, 70, 385, 3083, 3072, 40),
          (20, .5, .0038, 70, 385, 2960, 2940, 81), (23, .5, .0055, 70, 385, 2975, 2960, 56), (26, .5, .0076, 70, 385, 2864, 2846, 40),
          (3, .1, .0038, 90, 495, 2940, 2935, 63), (6, .1, .0055, 90, 495, 2825, 2820, 43), (9, .1, .0076, 90, 495, 2827, 2825, 31),
          (12, .3, .0038, 90, 495, 2820, 2806, 63), (15, .3, .0055, 90, 495, 2877, 2864, 43), (18, .3, .0076, 90, 495, 2780, 2766, 31),
          (21, .5, .0038, 90, 495, 2846, 2827, 63), (24, .5, .0055, 90, 495, 2913, 2895, 43), (27, .5, .0076, 90, 495, 2895, 2877, 31)]
FEED_LEVEL = {.0038: .10, .0055: .15, .0076: .20}   # nominal level used in the paper


def mrr_from_mass(dm_g, t_s):
    """MRR [mm^3/min] = dm [g] / rho [g/cm^3] * 1000 [mm^3/cm^3] / t [s] * 60 [s/min]"""
    return dm_g / RHO_TI64 * 1000.0 / t_s * 60.0


def dataset_E(dm_override=None):
    m = pd.DataFrame(E_MASS, columns=['run', 'ap', 'f_in', 'speed', 'N_rpm', 'm_before', 'm_after', 't_recorded'])
    m['f'] = m.f_in.map(FEED_LEVEL); m['f_mm'] = m.f_in * 25.4
    m['dm'] = m.m_before - m.m_after
    m['t_calc'] = 60.0 * CUT_LENGTH_E / (m.N_rpm * m.f_mm)
    if dm_override is not None: m['dm'] = dm_override(m)
    m['MRR'] = mrr_from_mass(m.dm, m.t_calc)
    r = pd.DataFrame(E_RA, columns=['group', 'speed', 'f', 'ap', 'Ra'])
    d = r.merge(m, on=['speed', 'f', 'ap'], how='left', validate='one_to_one')
    d = d.sort_values('group').reset_index(drop=True)
    d['source_row'] = d.group; d['category'] = ''; d['MRR_type'] = 'measured (mass loss)'
    return d


DATASETS = {'A': dataset_A, 'B': dataset_B, 'C': dataset_C, 'D': dataset_D, 'E': dataset_E}
EXPECTED_ROWS = {'A': 27, 'B': 18, 'C': 13, 'D': 9, 'E': 27}
SOURCES = {'A': 'Rajaparthiban & Naveen Sait (2018), Mechanika 24(4), Table 3',
           'B': 'Belbellaa et al. (2023), Periodica Polytechnica Mech. Eng. 67(1), Table 4',
           'C': 'Niharika et al. (2016), Archive of Mechanical Engineering 63(4), Box-Behnken results table',
           'D': 'Khan & Jaffery (2025), Procedia CIRP 135, Tables 1-2',
           'E': 'Ra: Priyadarsini et al. (2026), Eng. Res. Express 8 085409, Table 5; MRR: authors\' mass-loss records'}


def validate(key, d):
    """Input checks run before any ranking."""
    assert len(d) == EXPECTED_ROWS[key], f'{key}: expected {EXPECTED_ROWS[key]} rows, got {len(d)}'
    for c in ['speed', 'f', 'ap', 'Ra', 'MRR']:
        assert d[c].notna().all() and np.isfinite(d[c].astype(float)).all(), f'{key}: missing/non-finite {c}'
    assert (d.Ra > 0).all() and (d.MRR > 0).all(), f'{key}: non-positive Ra or MRR'
    keys = d[['speed', 'f', 'ap', 'category']].astype(str).agg('|'.join, axis=1)
    assert keys.is_unique, f'{key}: duplicate condition keys {keys[keys.duplicated()].tolist()}'
    assert d.source_row.is_monotonic_increasing, f'{key}: candidate order is not the source order'
    return True
