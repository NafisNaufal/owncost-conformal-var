# Amendment 2 to the frozen confirmatory protocol (written 2026-11-15, before any 2026 data was downloaded)

Applies to preregistration.md (sha256 35b669228f9f9db0f898ec72f4c50513bb7f59ca0a924f9469d71cff68ff90b2, unchanged) and
preregistration_amendment1.md (sha256 5080994a3df5344b04e0dfd76ec4d68be4b0b72fb54361301ca5316ff0ca24d1, unchanged). Status check before writing: the latest
date in all 31 source files is 2025-12-31 and all 31 checksums in data_checksums.sha256 verify (checked by script).
Reason: round-4 review (R6, R7, S11, S12, S13) of manuscript v5. Correction: amendment 1's reason line says the round-3
review was "of manuscript v5"; round 3 reviewed manuscript v4.

## Changes
1. H2 is descriptive, not a hypothesis test. Amendment 1 compared the learned buffer's first-day change with null draws built
   by a different procedure (the tracker's VaR times a fixed uplift on shifted-flag days), so the statistic and the null were
   not exchangeable; and a 715-day window should hold only about 8 onsets. The confirmatory report gives the US-led
   first-day change of the buffer minus the tracker, and, as a same-procedure contrast, the fixed uplift applied on the real
   flag against the same uplift on the 40 shifted flags, with draw counts. No decision is attached to either.
2. H1 decision rule, stated exactly (implemented in oc4_confirm.py). For each level, d_t is the mean over the 34
   target x base cells of the buffer's tick loss minus the tracker's on day t of the window, and dbar is the mean of d_t.
   p = (1 + #{b : mean(d*_b) - dbar <= dbar}) / 3001, from 3000 draws of a 21-day circular block bootstrap of the centred
   series (seed 2026). Holm over the two levels: with p(1) <= p(2), reject the first if dbar < 0 and p(1) <= 0.025, and only
   then the second if dbar < 0 and p(2) <= 0.05. This replaces every earlier wording of the rule, including amendment 1's
   "at most one draw as low", which applied to H2.
3. Power by simulation of the test itself (oc4_power_v6.py, power_v6.json), replacing the normal approximations of
   power.json and power_v5.json. With windows drawn from the 2015-2025 daily differences, which keeps the exploratory effect
   (likely an overstatement): Holm rejection probability 0.24 at 1% and 0.41 at 5% for 715 days (0.13 and 0.24 for 215;
   0.53 and 0.75 for 1500). With the same series centred at zero the test rejects in about 0.05% of windows: it is
   conservative for this left-skewed series.
4. H1 does not test the cross-market claim. The buffer with the own gap only, and the tuned same-feature baselines of
   oc4_parity_v6.py (settings in parity_v6.json, selected on 2007-2014), are reported descriptively against the buffer.
5. Frozen entry point oc4_confirm.py: the window is a parameter; it verifies the 31 frozen series against
   data_checksums.sha256; vintage rule: rows dated on or before 2025-12-31 always come from the frozen files and rows from
   2026-01-01 come only from a fresh download of the same vendors, kept in a separate directory; history is never
   re-downloaded or replaced. Checked on the exploratory window (2015-01-01 to 2026-01-01, no new data): it reproduces the
   published cells (-7.75% at 1%, -3.04% at 5%), with p = 0.030 and 0.008 (exploratory, not confirmatory).
6. Unchanged: code, settings, targets, timing, and the earliest run date (not before 715 trading days of new data).
   The protocol has still not been deposited with an external timestamped registry; the author should do so before any
   2026 data are downloaded.

## Hashes (sha256)
48f7b55483cf2d8f780b3911bb9a9c0c457b54099608c8ce2878f007dc5ed084  oc4_confirm.py
7a45aa66a39936e53ad29bf42cefad7d91c56f7cea96533f13a81931f78d01a4  oc4_power_v6.py
c2cc793890fa2939b1cd806ad17999f6158e9a9db9e409926391de1f9f7f579b  power_v6.json
a865a963a34b8995f79e5aa53a2e0603e79040551a36c871b3ea097b6772c222  oc4_parity_v6.py
f87fa7117f3d024c84e5aabaf59efba086c22f10c6b2d8aeb7196ee194e82dea  parity_v6.json
8fcef77406fd7ee988b9cd37552a362ea333feb188ac7525fb6a4b2db5f41fc9  oc4_v6.py
ecb6835f6920c74f2b6fb2890b32e6a9584b9769205b860a1713be526ba74a82  v6.json
5080994a3df5344b04e0dfd76ec4d68be4b0b72fb54361301ca5316ff0ca24d1  preregistration_amendment1.md
35b669228f9f9db0f898ec72f4c50513bb7f59ca0a924f9469d71cff68ff90b2  preregistration.md
9cc86f994d3c0fdc493a135edf168e3d6e3d6d51e36578a330e29469383d7f51  data_checksums.sha256
