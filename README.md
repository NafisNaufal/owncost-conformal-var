# Code, results and protocol for "The Per-Shock Price of Own Feedback in Online Conformal Value-at-Risk"

Companion archive for the paper "The Per-Shock Price of Own Feedback in Online Conformal Value-at-Risk: Lower Bounds,
Exact Per-Day Optima and Known-Law Attainment" (N. N. Rahman and T. M. Adn, Universitas Brawijaya; manuscript v7, November 2026).

## Contents
| Folder | What it holds |
|---|---|
| `paper/supplement.pdf` | Full proofs, simulations, exploratory-study details, the frozen protocol and the structured-review record |
| `code/theory/` | Simulations and numerical checks for the theory (Theorems 1-3, Proposition 1, Corollary 1: `finite_delta_v7.py`); `fig_theory.py` draws Figure 2 |
| `code/oc4/` | Exploratory empirical study: inputs (`oc4_build.py`), tuning (`oc4_study.py`), evaluation (`oc4_eval.py`), analyses, the event study, baseline parity, power, and the frozen confirmatory entry point `oc4_confirm.py`; `selected.pkl` and `placebo_selected.pkl` are the frozen settings tuned on 2007-2014 |
| `code/data_download/` | The scripts that downloaded the raw index data (Yahoo Finance via `yfinance`; Stooq files for Thailand and the Philippines were downloaded by hand and imported with `01b_manual_import.py`) |
| `results/` | JSON outputs of every script, as reported in the paper and supplement |
| `protocol/` | The frozen confirmatory protocol (12 November 2026), amendments 1-3 (15 November 2026), their SHA-256 files, and SHA-256 checksums of the 31 raw series; `results/oc4/h1_series.npz` is the frozen series used to calibrate H1 (amendment 3) |
| `figures/` | Figure 1 (overview, TikZ: `pdflatex fig_overview.tex`; bar data from `finite_delta_v7.json`) and Figure 2 |

## Data
The raw index series are not redistributed here, because they come from third-party vendors. Re-download them with the
scripts in `code/data_download/` (dates 2000-01-01 to 2025-12-31) and compare them against `protocol/data_checksums.sha256`.
Vendors revise history, so a fresh download may not match those checksums exactly; the exploratory results were computed
on files with exactly these checksums. Large intermediate files (`oc4_data.pkl`, about 160 MB; `cells.pkl`, about 2.2 GB)
are rebuilt by `oc4_build.py` and `oc4_eval.py`.

## Reproducing
Python 3.10 with the packages in `requirements.txt`. The scripts use absolute paths from the author's machine
(`/home/.../aciids/astar/...`); set them to your own layout before running. They were left unchanged because several are
hashed in the frozen protocol. Order: `code/data_download` -> `oc4_build.py` -> `oc4_study.py` -> `oc4_eval.py` ->
the analysis scripts (`oc4_analysis.py`, `oc4_event.py`, `oc4_v5.py`, `oc4_ownstress.py`, `oc4_parity_v6.py`, `oc4_v6.py`,
`oc4_v6b.py`, `oc4_v7.py`, `oc4_parity_v7.py`, `oc4_h1_series.py`, `oc4_power_v7.py`). Theory scripts run on their own (`sim_v7_match.py` and `finite_delta_v7.py` use about 200 cores for several minutes).

## Confirmatory test
Not yet run. It is to be run with `oc4_confirm.py` (amendment 3; the amendment-2 version is `oc4_confirm_a2.py`) once 715
trading days after 2025 are available, as the protocol and its amendments specify. Run on the exploratory window (`--start 2015-01-01 --end 2026-01-01 --new-data none`), it reproduces
the published cells (-7.75% at 1%, -3.04% at 5%).

## Status of the evidence
The empirical results are exploratory: the 2015-2025 window had been examined before the design was fixed. The proofs
have not been checked by anyone other than the author.

## Licence
Code (`code/`, and the TikZ source `figures/fig_overview.tex`) is released under the MIT licence (`LICENSE`). The supplement,
results, figures and protocol documents are released under CC BY 4.0 (`LICENSE-CC-BY-4.0`). Raw vendor data are not
included and are not covered by either licence.

## Before depositing (author)
- After deposit, add the DOI to the paper and to `CITATION.cff`.
