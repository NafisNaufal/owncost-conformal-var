"""v5: adds power at the first Holm step (one-sided 2.5%) for the two-level family. Power of a confirmatory test of LinBuf vs Tracker (pooled tick loss) on a window of n trading days, from the 2015-2025 daily
pooled differences (21-day block bootstrap of window means; effect set to the exploratory estimate, which likely overstates it)."""
import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/oc4')
import json, pickle
import numpy as np
from scipy.stats import norm
import oc4_study as S
import oc4_analysis_lite as A
R = pickle.load(open('/home/nafisnaufal1426/aciids/astar/oc4/cells.pkl', 'rb')); rng = np.random.default_rng(5); out = []
for a in S.LV:
    d = A.daily(R, a, 'LinBuf') - A.daily(R, a, 'Tracker'); dm = np.nanmean(d, 1); dm = dm[np.isfinite(dm)]; T = len(dm); mu = dm.mean()
    for n in [215, 465, 715]:
        means = []
        for _ in range(4000):
            st = rng.integers(0, T, int(np.ceil(n / 21))); idx = ((st[:, None] + np.arange(21)).ravel()[:n]) % T; means.append(dm[idx].mean())
        se = np.std(means); z = norm.ppf(0.975)
        out.append(dict(alpha=a, n_days=n, effect=100 * mu, se=100 * se, power_two_sided_5pct=float(norm.cdf(-mu / se - z) + norm.cdf(mu / se - z)), power_one_sided=float(norm.cdf(-mu / se - norm.ppf(0.95))), power_holm_first_step=float(norm.cdf(-mu / se - norm.ppf(0.975)))))
for r in out: print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
json.dump(out, open('/home/nafisnaufal1426/aciids/astar/oc4/power_v5.json', 'w'), indent=1)
