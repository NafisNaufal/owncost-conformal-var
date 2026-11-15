# Amendment 3 to the frozen confirmatory protocol (written 2026-11-15, before any 2026 data was downloaded)

Applies to preregistration.md and amendments 1 and 2 (hashes below, unchanged). Status check before writing: the latest
date in all 31 source files is 2025-12-31 and all 31 checksums in data_checksums.sha256 verify (checked by script).
Reason: round-5 review (R4, S17) of manuscript v6.

## Changes
1. H1 calibration. Under the null, the amendment-2 test rejects in about 0.05% of windows, because the one-sided block-bootstrap
   p-value of the mean of a strongly left-skewed series is far from uniform (power_v6.json; null 2.5% quantile of p about
   0.11 at 715 days). The statistic and the raw p-value are unchanged. The decision now uses a calibrated p-value:
   p_cal = (1 + #{j : p0_j <= p}) / (J + 1), where p0_1..p0_J (J = 2000) are raw p-values of windows of the same length n
   drawn as 21-day circular blocks from the frozen 2015-2025 daily differences in h1_series.npz, centred at zero, each
   computed with the same bootstrap (seeds 10000 + j). Holm over the two levels on p_cal, with thresholds 0.025 and 0.05 and
   dbar < 0 required, as in amendment 2. Implemented in oc4_confirm.py; the amendment-2 version is kept as oc4_confirm_a2.py.
2. Assumption. The calibration assumes that under the null the daily differences have the shape of the 2015-2025 series,
   shifted to mean zero. If the confirmatory window has a different shape, the calibrated size can differ from nominal.
3. Reporting. The output also gives the raw p, the null quantiles of p (1%, 2.5%, 5%, 50%) and the null probability of
   dbar < 0, and, for every post-2025 input file, its SHA-256, date range and number of rows.
4. Size and power (oc4_power_v7.py, power_v7.json; 2000 windows each). Per-level Holm rejection under the null:
   0.034/0.040 (215 days), 0.023/0.027 (715), 0.031/0.030 (1500) at 1%/5%. With the exploratory effect (likely an
   overstatement): 0.60/0.64 (215), 0.79/0.87 (715), 0.96/0.99 (1500). The amendment-2 rule on the same windows: size
   at most 0.001, power 0.21/0.38 at 715 days.
5. Check. On the exploratory window with no new data, oc4_confirm.py reproduces -7.75% (1%) and -3.04% (5%), raw p 0.030
   and 0.008. The calibrated p (0.002 and 0.0005) is circular there, because the calibration series is the same data.
6. Unchanged: hypotheses, code that builds the inputs and runs the methods (oc4_lib.py, oc4_build.py, oc4_study.py),
   settings (selected.pkl), targets, timing, and the earliest run date (not before 715 trading days of new data).
   The protocol and its amendments have still not been deposited with an external timestamped registry; the author
   should do so before any 2026 data are downloaded.

## Hashes (sha256)
974328490bbb5f9a1188f7828b53631da277437c0a7a56c3ea18ee13a725f9b8  oc4_confirm.py
48f7b55483cf2d8f780b3911bb9a9c0c457b54099608c8ce2878f007dc5ed084  oc4_confirm_a2.py
9bfe3dc8500f010a4181d99f9710be22bbbbfd8066e65fbf2e96f220da5613b3  h1_series.npz
91fc6fbab41021038d80df4cb05c23a283f3fab2c7e8439a16e91cf3b92ef97a  oc4_h1_series.py
97a0516cd8c1d230602ec4d1436fd503852f17383cc60a9d875c76ca93ea25e9  oc4_power_v7.py
5f1434ee2bff32a94f58a504c5101d5eae2fd34d7aa160664ed52039bf6da435  power_v7.json
6e7be1cd4615a2276b56adae6d18b1a6f5d6a46adfdd28b58f4b217ce27d687b  oc4_v7.py
35c14001fa2f8fc90a49600ecf221341ed20b720733c0afcb6efb97d3a10a00a  v7.json
fde2a2b595fb39414792c9ade5750c2f9567da0c0c1324d181dbe14fbcfff96c  oc4_parity_v7.py
eb1dfb03c16c144040f72b3906deb214bd83612c311f48c1b1fe1816c0a83441  parity_v7.json
35b669228f9f9db0f898ec72f4c50513bb7f59ca0a924f9469d71cff68ff90b2  preregistration.md
5080994a3df5344b04e0dfd76ec4d68be4b0b72fb54361301ca5316ff0ca24d1  preregistration_amendment1.md
d0146de3295451d0fcfe8700ea951d94489ba767f4feaeabf4fa3e6f22269d5a  preregistration_amendment2.md
9cc86f994d3c0fdc493a135edf168e3d6e3d6d51e36578a330e29469383d7f51  data_checksums.sha256
8282d5e083100ec9f0e4f28de5a2cbb7a899001bd551b78524f3f483322b9703  oc4_lib.py
0dd1ee8e8848004a624de8ca9f0442c515c7e467b217b8832e8e28076616bbba  oc4_build.py
38eda206cc2832ee3cb57fb2d08a1cfd8b893f2334e53f0b65bcac47284eca05  oc4_study.py
861407ed959f81ca3b5bcd0a51116c8250e6289e0c85a7df2e01685669fb581b  selected.pkl
