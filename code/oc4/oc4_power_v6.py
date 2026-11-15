"""v6 (round-4 S11): power of the frozen H1 test, simulated with the test itself (oc4_confirm.h1_test and holm) rather than a
normal approximation, and its size under the null (the same series centred at zero). Windows of n trading days are drawn as 21-day circular blocks of the 2015-2025 pooled daily differences
(the same days for both levels, so the Holm family is simulated jointly), which keeps the exploratory effect (likely an
overstatement). 2000 simulated windows per n, 3000 inner bootstrap draws each. Writes power_v6.json."""
import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/oc4')
import json, pickle
import numpy as np
from multiprocessing import Pool
import oc4_study as S
import oc4_analysis_lite as A
from oc4_confirm import h1_test, holm
R = pickle.load(open('/home/nafisnaufal1426/aciids/astar/oc4/cells.pkl', 'rb'))
DD = {a: np.nanmean(A.daily(R, a, 'LinBuf') - A.daily(R, a, 'Tracker'), 1) for a in S.LV}
ok = np.isfinite(DD[0.01]) & np.isfinite(DD[0.05]); DD = {a: v[ok] for a, v in DD.items()}; T = int(ok.sum()); del R
DN = {a: v - v.mean() for a, v in DD.items()}


def sim(t):
    n, seed, null = t; X = DN if null else DD; rng = np.random.default_rng(seed); st = rng.integers(0, T, int(np.ceil(n / 21))); idx = ((st[:, None] + np.arange(21)).ravel()[:n]) % T
    res = {a: h1_test(X[a][idx], seed=seed)[:2] for a in S.LV}; rj = holm(res)
    return n, null, rj[0.01], rj[0.05], res[0.01][1] <= 0.025 and res[0.01][0] < 0, res[0.05][1] <= 0.025 and res[0.05][0] < 0


if __name__ == '__main__':
    with Pool(200) as p:
        out = p.map(sim, [(n, 1000 * i + n, null) for null in [False, True] for n in [215, 465, 715, 1500] for i in range(2000)])
    rows = []
    for null in [False, True]:
      for n in [215, 465, 715, 1500]:
        z = np.array([o[2:] for o in out if o[0] == n and o[1] == null], float)
        rows.append(dict(kind='size (null)' if null else 'power (exploratory effect)', n_days=n, holm_reject_1pct=float(z[:, 0].mean()), holm_reject_5pct=float(z[:, 1].mean()), any_reject=float((z[:, 0] + z[:, 1] > 0).mean()),
                         first_step_1pct=float(z[:, 2].mean()), first_step_5pct=float(z[:, 3].mean())))
    for r in rows: print(r)
    json.dump(rows, open('/home/nafisnaufal1426/aciids/astar/oc4/power_v6.json', 'w'), indent=1)
