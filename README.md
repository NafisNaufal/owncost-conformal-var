# Code, results and protocol for "What Own Feedback Costs Online Conformal Value-at-Risk"

Companion archive for the paper "What Own Feedback Costs Online Conformal Value-at-Risk: Per-Shock Lower Bounds,
Known-Law Attainment and the Exact First Day" (N. Naufal, Universitas Brawijaya; manuscript v6, November 2026).

## Contents
| Folder | What it holds |
|---|---|
| `paper/supplement.pdf` | Full proofs, simulations, exploratory-study details, the frozen protocol and the structured-review record |
| `code/theory/` | Simulations and numerical checks for the theory (Theorems 1-3, Proposition 1); `fig_theory.py` draws Figure 1 |
| `code/oc4/` | Exploratory empirical study: inputs (`oc4_build.py`), tuning (`oc4_study.py`), evaluation (`oc4_eval.py`), analyses, the event study, baseline parity, power, and the frozen confirmatory entry point `oc4_confirm.py`; `selected.pkl` and `placebo_selected.pkl` are the frozen settings tuned on 2007-2014 |
| `code/data_download/` | The scripts that downloaded the raw index data (Yahoo Finance via `yfinance`; Stooq files for Thailand and the Philippines were downloaded by hand and imported with `01b_manual_import.py`) |
| `results/` | JSON outputs of every script, as reported in the paper and supplement |
| `protocol/` | The frozen confirmatory protocol (12 November 2026), amendments 1 and 2 (15 November 2026), their SHA-256 files, and SHA-256 checksums of the 31 raw series |
| `figures/` | Figure 1 |

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
`oc4_v6b.py`, `oc4_power_v6.py`). Theory scripts run on their own (`sim_v6_match.py` uses about 200 cores for a few minutes).

## Confirmatory test
Not yet run. It is to be run with `oc4_confirm.py` once 715 trading days after 2025 are available, as the protocol and its
amendments specify. Run on the exploratory window (`--start 2015-01-01 --end 2026-01-01 --new-data none`), it reproduces
the published cells (-7.75% at 1%, -3.04% at 5%).

## Status of the evidence
The empirical results are exploratory: the 2015-2025 window had been examined before the design was fixed. The proofs
have not been checked by anyone other than the author.

## Before depositing (author)
- Choose a licence (for example MIT for code and CC BY 4.0 for documents) and add a LICENSE file; none is included yet.
- Add the GitHub URL and, after deposit, the DOI to the paper and to `CITATION.cff`.
