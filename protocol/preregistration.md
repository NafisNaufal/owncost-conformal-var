# Frozen protocol for a confirmatory test (written 2026-11-12, before any 2026 data was downloaded)

Status: frozen. No data dated 2026-01-01 or later was downloaded or inspected when this file and its hashes were written (the
latest date in every source file is 2025-12-31 or earlier; checked by script before writing).

## Design (unchanged from the exploratory v4 study)
- Code: oc4_lib.py, oc4_build.py, oc4_study.py (hashes below). Settings: selected.pkl (tuned on 2007-2014; hash below).
  No re-tuning. Targets: the 17 indices of the v4 study. Base models: EWMA-normal and HS-250. Timing: VaR at the local open,
  open-to-close loss, inputs time-stamped before the open (time-zone-aware clocks, conservative closes, age <= 3 days).
- Data: daily OHLC from the same vendors (Yahoo Finance; Stooq for THA and PHL), from 2026-01-01.

## Hypotheses (one-sided)
- H1 (primary): pooled tick loss of the learned buffer is lower than the quantile tracker's, at alpha = 5% and at alpha = 1%.
  Test: 21-day circular block bootstrap of the pooled daily difference (3000 draws); one-sided p-values; Holm over the two levels at 5%.
- H2 (secondary): at shock onsets defined exactly as in oc4_event.py (Q = 20, c = 0.25, w = 1), the first-day violation rate of
  the buffer minus the tracker is negative in the US-led group, at both levels; wild cluster bootstrap over episodes; reported
  with the self-grouped null of oc4_event.py.
- Everything else is descriptive.

## When to run
- Not before 715 trading days of new data are available (about end of 2028). Power for H1, assuming the exploratory effect
  (likely an overstatement): 715 days -> one-sided power about 0.30 (1%) and 0.45 (5%); 215 days -> 0.15 and 0.21 (power.json).
  A null result at that sample size is weak evidence against H1, and will be reported as such.

## Hashes (sha256)
8282d5e083100ec9f0e4f28de5a2cbb7a899001bd551b78524f3f483322b9703  oc4_lib.py
0dd1ee8e8848004a624de8ca9f0442c515c7e467b217b8832e8e28076616bbba  oc4_build.py
38eda206cc2832ee3cb57fb2d08a1cfd8b893f2334e53f0b65bcac47284eca05  oc4_study.py
fffa907757cfb9169163ec806d569e8ef03fe1ae02ceb953df9a4b639921b37e  oc4_event.py
861407ed959f81ca3b5bcd0a51116c8250e6289e0c85a7df2e01685669fb581b  selected.pkl
62090eb06a395c21b63da6727f879fbea95a17756803c4aeeb7e0133286c6d76  power.json
