"""Numerical check of Theorem 3 (v5): known-law Shiryaev-Roberts GCT on the own-score stream with the rate conditions of (c):
gamma = r H1 / delta, r = m + 10, eta0 = delta log(1/delta), eta1 = 0.5, gaps much longer than 1/eta0. alpha = 5%, calm N(0,1),
shock N(0, sigma^2), m = 60. Reports realised calm cost against delta, mean pre-alarm shock days (1 + delay) against the
first-order log(1/delta)/D_P, and excess violations per burst over alpha m against (p0 - alpha) times the first-order days."""
import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/theory')
import json
import numpy as np
from multiprocessing import Pool
from scipy.stats import norm
import sim_gct as G
A = 0.05; q0 = norm.ppf(1 - A)
El = lambda q: (1 - A) * (norm.pdf(q) - q * norm.sf(q)) + A * (q * norm.cdf(q) + norm.pdf(q))
h0 = lambda q: El(q) - El(q0); l0 = El(q0)


def job(t):
    seed, sig, m, delta, g, ncyc = t; r = m + 10; H1 = float(h0(sig * q0)); gamma = r * H1 / delta; eta0 = delta * np.log(1 / delta)
    out = G.run_path(seed, A, sig, m, g, 0, 0, 0, gamma, r, eta0, 0.5, ncyc, G.KAPPAS, True, True)
    return t, out


if __name__ == '__main__':
    rows = []; tasks = []
    for sig in [2.0, 3.0]:
        for delta in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
            eta0 = delta * np.log(1 / delta); g = int(max(2000, 15 / eta0)); ncyc = 12
            tasks += [(s, sig, 60, delta, g, ncyc) for s in range(96)]
    with Pool(200) as p:
        res = p.map(job, tasks)
    for sig in [2.0, 3.0]:
        DP = 0.5 * (sig ** 2 - 1 - np.log(sig ** 2)); p0 = norm.sf(q0 / sig)
        for delta in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
            rr = [o for (t, o) in res if t[1] == sig and t[3] == delta]
            cc = sum(o[0] for o in rr) / sum(o[1] for o in rr) - l0
            V = np.concatenate([o[2] for o in rr]); ad = np.concatenate([o[8] for o in rr]); det = ad >= 0
            pre = np.where(det, np.minimum(ad + 1, 60), 60)  # pre-alarm shock days (alarm day counted as day ad; undetected = whole burst)
            first = np.log(1 / delta) / DP
            rows.append(dict(sigma=sig, delta=delta, eta0=delta * np.log(1 / delta), calm_cost=float(cc), calm_over_delta=float(cc / delta),
                             first_order_days=float(first), first_order_days_at_realised_cost=float(np.log(1 / cc) / DP), mean_pre_alarm_days=float(pre.mean()),
                             ratio_days=float(pre.mean() / (np.log(1 / cc) / DP)), detected=float(det.mean()), V=float(V.mean()), excess=float(V.mean() - A * 60),
                             excess_first_order=float((p0 - A) * min(60, np.log(1 / cc) / DP))))
    for r in rows: print({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    json.dump(rows, open('/home/nafisnaufal1426/aciids/astar/theory/sim_v5_match.json', 'w'), indent=1)
