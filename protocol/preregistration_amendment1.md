# Amendment 1 to the frozen confirmatory protocol (written 2026-11-15, before any 2026 data was downloaded)

Applies to preregistration.md (sha256 35b669228f9f9db0f898ec72f4c50513bb7f59ca0a924f9469d71cff68ff90b2, unchanged).
Status check before writing: the latest date in all 31 source files is 2025-12-31 (checked by script; checksums in
data_checksums.sha256). Reason: round-3 review (R8, S12, S13) of manuscript v5.

## Changes
1. H2 null replaced. The self-grouped null of oc4_event.py removes the buffer's VaR uplift together with the information.
   H2 is now tested against the uplift-matched null of oc4_v5.py ("uplift" block): for each of the 40 common lags in
   oc4_eval.LONG, the tracker's VaR is multiplied, on days the shifted US flag is on, by the buffer's own flag-day uplift
   (ratio of mean VaRs on real-flag days, per target x base x level, computed on the confirmatory window), and onsets
   are grouped by the shifted flag. H2 holds at a level if the real US-led first-day change is negative and
   p = (1 + #{null draws <= real}) / 41 <= 0.05 (at most one draw as low), Holm over the two levels.
   Note: on the exploratory 2015-2025 data this amended H2 would not hold (3 of 40 draws at 1%, 32 of 40 at 5%).
2. Power for H1 is reported at the first Holm step (one-sided 2.5%): about 0.20 (1%) and 0.33 (5%) at 715 days,
   0.09 and 0.13 at 215 days (power_v5.json). The original file's 0.30/0.45 were one-sided 5% per level.
3. Additional hashes: analysis code, placebo code and settings, the v5 additions, and the raw input series.
4. Everything else is unchanged. The protocol has not been deposited with an external timestamped registry; the author
   should do so (e.g. OSF) before the confirmatory run.

## Hashes (sha256)
9f60356ba54bcdecbc5056fabf469289d687f50d03ae6abdb0146021d8c476a6  oc4_analysis.py
98d64bc32baa55038e05bea459bcbe8dbccffac9213554d640c4f8decb3ea2f8  oc4_eval.py
e52319c79696ac86970fae474ad7cea48cafcc523045241ddceb7a08964e79db  placebo_selected.pkl
fffa907757cfb9169163ec806d569e8ef03fe1ae02ceb953df9a4b639921b37e  oc4_event.py
8f30502526f9cd7bcc84c2fe9893ffa1c58bb75354191aebd47e3d3d4a6446d6  oc4_v5.py
351867034fd526ba97d9a8d6ed338a20cb8b318156eebe5af5f46596a04ee947  oc4_ownstress.py
b20229537216262a8c230ebc1203319f42d160cda7152742281a49109633ca4b  oc4_power_v5.py
3df0ef3517d29a764516b965d99d9c244140b9e28f1de0c66f1ebdbb440f369c  power_v5.json
9cc86f994d3c0fdc493a135edf168e3d6e3d6d51e36578a330e29469383d7f51  data_checksums.sha256
35b669228f9f9db0f898ec72f4c50513bb7f59ca0a924f9469d71cff68ff90b2  preregistration.md
