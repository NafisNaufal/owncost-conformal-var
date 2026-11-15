"""v7 (round-5 R4): size and power of the calibrated H1 test of amendment 3 (oc4_confirm.py), simulated with the test itself.
For each window length n: J = 2000 null raw p-values per level (oc4_confirm.null_raw_p, seeds 10000 + j) define the
calibration; then 2000 further windows (different seeds) are drawn either from the centred series (size) or from the raw
2015-2025 series, which keeps the exploratory effect and likely overstates it (power), with the same days for both levels.
Each window is tested with the raw bootstrap p, calibrated, and Holm. Also reports the amendment-2 (uncalibrated) rule on
the same windows. Writes power_v7.json."""
import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/oc4')
import json
import numpy as np
from multiprocessing import Pool
from oc4_confirm import h1_test, holm, calibrate
Z = np.load('/home/nafisnaufal1426/aciids/astar/oc4/h1_series.npz'); DD = {0.01: Z['d01'], 0.05: Z['d05']}; T = len(DD[0.01])
DC = {a: v - v.mean() for a, v in DD.items()}


def nullp_job(t):
    n, a, j = t; rng = np.random.default_rng(10000 + j); st = rng.integers(0, T, int(np.ceil(n / 21))); idx = ((st[:, None] + np.arange(21)).ravel()[:n]) % T
    return t, h1_test(DC[a][idx], seed=10000 + j)[1]


def win_job(t):
    n, kind, j = t; X = DC if kind == 'null' else DD; rng = np.random.default_rng(500000 + 7 * j + (0 if kind == 'null' else 3))
    st = rng.integers(0, T, int(np.ceil(n / 21))); idx = ((st[:, None] + np.arange(21)).ravel()[:n]) % T
    return t, {a: h1_test(X[a][idx], seed=600000 + j)[:2] for a in (0.01, 0.05)}


if __name__ == '__main__':
    NS = [215, 715, 1500]; J = 2000; W = 2000
    with Pool(200) as p:
        NP = dict(p.map(nullp_job, [(n, a, j) for n in NS for a in (0.01, 0.05) for j in range(J)], chunksize=20))
        WR = dict(p.map(win_job, [(n, k, j) for n in NS for k in ('null', 'alt') for j in range(W)], chunksize=20))
    rows = []
    for n in NS:
        nullp = {a: np.array([NP[(n, a, j)] for j in range(J)]) for a in (0.01, 0.05)}
        for kind in ('null', 'alt'):
            rc = {0.01: 0, 0.05: 0}; ru = {0.01: 0, 0.05: 0}
            for j in range(W):
                r = WR[(n, kind, j)]
                cal = holm({a: (r[a][0], calibrate(r[a][1], nullp[a])) for a in r}); unc = holm({a: r[a] for a in r})
                for a in r: rc[a] += cal[a]; ru[a] += unc[a]
            rows.append(dict(n_days=n, kind='size (null)' if kind == 'null' else 'power (exploratory effect)',
                             calibrated_reject_1pct=rc[0.01] / W, calibrated_reject_5pct=rc[0.05] / W, uncalibrated_reject_1pct=ru[0.01] / W, uncalibrated_reject_5pct=ru[0.05] / W,
                             null_p_q025_1pct=float(np.quantile(nullp[0.01], 0.025)), null_p_q025_5pct=float(np.quantile(nullp[0.05], 0.025))))
            print(rows[-1], flush=True)
    json.dump(rows, open('/home/nafisnaufal1426/aciids/astar/oc4/power_v7.json', 'w'), indent=1)
