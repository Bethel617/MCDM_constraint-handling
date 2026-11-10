# Feasibility filtering versus penalties in multi-criteria selection of process conditions — data and code

Data and code for the article *"When does constraint handling change the decision? Feasibility filtering versus penalties in multi-criteria selection of process conditions"* (submitted to *Computers & Industrial Engineering*).

The repository contains the five Ti-6Al-4V turning datasets used in the study, the analysis code that regenerates every table, figure and number in the article, the outputs it produces, and an independent verification script.

## Quick start

    pip install -r requirements.txt
    python run_all.py              # ~25 s, plus ~4 min for the synthetic study if it is recomputed; writes ./out/
    python verify_independent.py   # ~15 s; prints ALL CHECKS PASSED

The synthetic study is cached in `out/synthetic.csv.gz`; set the environment variable `RERUN_SYNTHETIC=1` to recompute it.
Tested with Python 3.13, numpy 2.x, pandas 2.x, matplotlib 3.x.

## Data (`data/`)

| File | Content | Source |
|---|---|---|
| `dataset_A.csv` | 27 runs, L27 design, ceramic-coated insert; MRR measured | Rajaparthiban & Naveen Sait (2018), *Mechanika* 24(4), 449–455, Table 3 |
| `dataset_B.csv` | 18 runs, L18 design, MQL; uncoated H13A and PVD-coated GC1125 inserts; nominal MRR | Belbellaa et al. (2023), *Periodica Polytechnica Mechanical Engineering* 67(1), 1–11, Table 4 |
| `dataset_C.csv` | 13 conditions, Box–Behnken design, wet, PVD-coated carbide; nominal MRR | Niharika et al. (2016), *Archive of Mechanical Engineering* 63(4), results table |
| `dataset_D.csv` | 9 runs, L9 design, dry / wet / cryogenic; nominal MRR | Khan & Jaffery (2025), *Procedia CIRP* 135, 538–542, Tables 1–2 |
| `dataset_E.csv` | 27 conditions, 3³ factorial, dry, coated tungsten carbide; Ra from the cited article, MRR from workpiece mass-loss records (mass before/after, cutting time) | Ra: Priyadarsini et al. (2026), *Engineering Research Express* 8, 085409, Table 5; mass-loss records: this study |

Columns: `speed` cutting speed (m/min; dataset A spindle speed, rpm), `f` feed (mm/rev), `ap` depth of cut (mm), `Ra` (µm), `MRR` (mm³/min), `MRR_type` (measured or nominal = 1000·Vc·f·ap), `category` (tool grade for B, cooling environment for D), `source_row` (row order in the source table, used as the candidate order). Dataset E also gives the spindle speed `N_rpm`, the actual feed `f_mm`, the masses (g), the mass loss `dm` (g, 1 g balance resolution), the recorded and calculated cutting times (s).
Datasets A–D are transcribed from the published tables listed above; please cite the original articles when reusing them. The same data are defined, with input checks, in `data_sources.py`.

## Code

| File | Content |
|---|---|
| `data_sources.py` | Input data with source-table identifiers and validation |
| `methods.py` | TOPSIS, VIKOR, PROMETHEE II, SAW, MOORA (definitions in Appendix B of the article) |
| `core.py` | FFR, PER (multiplicative and additive), FFR-fixed and the closed-form threshold λ* (Proposition 2) |
| `extended.py` | Empirical analysis over all methods and penalty forms; synthetic study |
| `run_all.py` | Main script: reference analysis, consequences, dataset E mass-resolution check, extended analysis and all figures |
| `make_figures.py`, `ext_figures.py` | Figures 1–6 |
| `verify_independent.py` | Re-implements the methods and decision rules in plain Python without the analysis code, recomputes all 41,310 empirical decisions by brute force, re-solves 3,000 random synthetic configurations and checks the case counts reported in the article |

## Outputs (`out/`)

| File | Content |
|---|---|
| `consolidated_datasets.csv` | Table 2 and Table A.1 data |
| `cases.csv` | TOPSIS-multiplicative decision cases (all runs and Pareto-only) with winners, outcome, λ*, tie flags |
| `table_outcomes_counts.csv`, `table_outcomes_totals.csv` | Outcome counts by dataset and candidate set (Table 3) |
| `ext_cases.csv` | Every empirical case for all methods and penalty forms (41,310 rows; Table 4) |
| `ext_thresholds.csv` | λ* per limit, weight, method and penalty form (Table 5) |
| `synthetic.csv.gz` | Synthetic study (2,754,000 configurations) |
| `consequences_cases.csv`, `table_consequences.csv` | Consequences of disagreement (Table 6, Fig. 6) |
| `E_mass_resolution_draws.csv` | Dataset E mass-resolution check (500 draws, seed 1) |
| `summary.json`, `ext_log.txt`, `run_log.txt` | Other numbers quoted in the text |
| `Fig1`–`Fig6` `*.png` | Figures 1–6 |

## Conventions

Candidate order = source-table order; argmax ties go to the first candidate (tolerance 1e-12, flagged in outputs).
Feasibility Ra ≤ T (inclusive); relative violation v = max(0, Ra − T)/T; T = `numpy.percentile(..., method='linear')` at 10, 15, …, 90 of each dataset's Ra.
Penalties: multiplicative s·exp(−λv), compared in log space (log s − λv), for methods with scores in [0, 1]; additive s − λv for all methods.
λ ∈ {0.5, 1, 2, 5, 10, 20, 50, 100, 1000}; w_Ra ∈ {0.3, 0.5, 0.7}.
TOPSIS closeness = 1 when d⁺ + d⁻ = 0. VIKOR v = 0.5, score 1 − Q, acceptability conditions not applied.
PROMETHEE II usual criterion, score (φ + 1)/2. SAW min–max normalisation. MOORA ratio system (additive penalty only).

## Licence

Code: MIT licence (`LICENSE`). Data and outputs (`data/`, `out/`): CC BY 4.0 (`LICENSE-DATA`); datasets A–D remain attributable to their original publications.

## Citation

Please cite the article and this repository (see `CITATION.cff`).
