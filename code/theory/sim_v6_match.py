"""v6 numerical check of Theorem 3(c) and the new 3(d) (round-4 R10, S7, S16).
Known-law Shiryaev-Roberts GCT on the own-score stream, rate conditions of Theorem 3: gamma = r H / delta, r = m + 10,
eta0 = delta log(1/delta), gaps max(2000, 15/eta0). alpha = 5%, calm N(0,1), bursts N(0, sigma^2) of m = 60 days at the end
of each cycle. Two holds:
  nominal      - state-1 quantile tracker (eta1 = 0.5), H = h0(sigma q0)   [Theorem 3(c)]
  conservative - state-1 threshold fixed at B = 4 sigma (shock breach rate 3e-5), H = h0(B)   [Theorem 3(d)]
Corrections against v5 (sim_v5_match.py): the alarm is checked before the threshold is set, so the alarm day is already
in state 1 and the number of state-0 (pre-alarm) shock days is ad, not ad + 1 (v5 line 36 used ad + 1).
Calm cost is measured on calm days more than r + 5 days after the previous burst (so post-burst holds, which are not
calm-law costs, are excluded) and more than 30 days before the next. Reports, per (sigma, delta, hold): realised calm cost delta', mean pre-alarm shock days N0 with path-level Monte Carlo
standard error, N0 - log(1/delta')/D_P, total violations per burst V with s.e., the first-order value
p0 min{m, log(1/delta')/D_P} and the ratio V / first-order, plus a fitted slope of N0 on log(1/delta') with a path
bootstrap interval on log(gamma) = log(1/delta) + log(r H) (first-order slope 1/D_P). Writes sim_v6_match.json."""
import json
import numpy as np
from numba import njit
from multiprocessing import Pool
from scipy.stats import norm
A = 0.05; q0n = norm.ppf(1 - A); M = 60; NP = 96; NCYC = 12
DELTAS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6]; SIGS = [2.0, 3.0]
El = lambda q: (1 - A) * (norm.pdf(q) - q * norm.sf(q)) + A * (q * norm.cdf(q) + norm.pdf(q))
h0 = lambda q: El(q) - El(q0n); l0 = El(q0n)


@njit(cache=True)
def erfc(x):
    z = abs(x); t = 1.0 / (1.0 + 0.5 * z)
    r = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return r if x >= 0 else 2.0 - r


@njit(cache=True)
def run(seed, a, sig, m, g, gamma, r, eta0, eta1, ncyc, qhold):
    """qhold > 0: conservative hold at qhold (no state-1 tracker); qhold <= 0: nominal state-1 tracker."""
    np.random.seed(seed)
    q0 = 1.6448536269514722; q1 = q0; R = 0.0; hold = 0; sprev = 0.0; T = g * ncyc
    calm_h = 0.0; calm_n = 0; V = np.zeros(ncyc); N0 = np.zeros(ncyc); det = np.zeros(ncyc)
    for t in range(T):
        cyc = t // g; pos = t % g; tau_pos = g - m; inburst = pos >= tau_pos
        alarm_state = hold > 0
        if not alarm_state:
            x = sprev * sprev
            R = sig ** (-1.0) * np.exp(0.5 * x * (1.0 - 1.0 / sig ** 2)) * (R + 1.0)
            if R >= gamma:
                hold = r; alarm_state = True
                if inburst and det[cyc] == 0:
                    det[cyc] = 1
        thr = (qhold if qhold > 0 else q1) if alarm_state else q0
        s = np.random.randn() * (sig if inburst else 1.0)
        e = 1.0 if s > thr else 0.0
        if t >= 2 * g:
            if inburst:
                V[cyc] += e
                if not alarm_state:
                    N0[cyc] += 1.0
            elif pos < tau_pos - 30 and pos > r + 5:
                z = thr; pdf = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi); sf = 0.5 * erfc(z / np.sqrt(2.0)); es = pdf - z * sf
                calm_h += (1 - a) * es + a * (z + es); calm_n += 1
        if alarm_state:
            if qhold <= 0:
                q1 += eta1 * (e - a)
        else:
            q0 += eta0 * (e - a)
        sprev = s
        if hold > 0:
            hold -= 1
            if hold == 0:
                R = 0.0
    return calm_h, calm_n, V[2:], N0[2:], det[2:]


def job(t):
    seed, sig, delta, kind = t; r = M + 10; eta0 = delta * np.log(1 / delta); g = int(max(2000, 15 / eta0))
    qhold = 4 * sig if kind == 'conservative' else -1.0
    H = float(h0(qhold)) if kind == 'conservative' else float(h0(sig * q0n)); gamma = r * H / delta
    return t, run(seed, A, sig, M, g, gamma, r, eta0, 0.5, NCYC, qhold)


if __name__ == '__main__':
    tasks = [(s, sig, d, k) for sig in SIGS for d in DELTAS for k in ['nominal', 'conservative'] for s in range(NP)]
    with Pool(200) as p:
        res = p.map(job, tasks)
    rng = np.random.default_rng(6); rows = []; slopes = []
    for sig in SIGS:
        DP = 0.5 * (sig ** 2 - 1 - np.log(sig ** 2)); p0 = float(norm.sf(q0n / sig))
        for kind in ['nominal', 'conservative']:
            per = {}
            for d in DELTAS:
                rr = sorted([(t[0], o) for (t, o) in res if t[1] == sig and t[2] == d and t[3] == kind])
                ch = np.array([o[0] for _, o in rr]); cn = np.array([o[1] for _, o in rr])
                Vp = np.array([np.mean(o[2]) for _, o in rr]); Np = np.array([np.mean(o[3]) for _, o in rr]); Dp = np.array([np.mean(o[4]) for _, o in rr])
                cc = ch.sum() / cn.sum() - l0; fo_days = np.log(1 / cc) / DP; fo_V = p0 * min(M, fo_days)
                per[d] = (ch, cn, Np)
                rows.append(dict(sigma=sig, hold=kind, delta=d, eta0=d * np.log(1 / d), calm_cost=float(cc), calm_over_delta=float(cc / d),
                                 N0=float(Np.mean()), N0_se=float(Np.std(ddof=1) / np.sqrt(len(Np))), first_order_days=float(fo_days),
                                 N0_minus_first=float(Np.mean() - fo_days), V=float(Vp.mean()), V_se=float(Vp.std(ddof=1) / np.sqrt(len(Vp))),
                                 first_order_V=float(fo_V), V_over_first=float(Vp.mean() / fo_V), excess=float(Vp.mean() - A * M),
                                 excess_over_first=float((Vp.mean() - A * M) / ((p0 - A) * min(M, fo_days))), detected=float(Dp.mean())))
            # slope of N0 on log(gamma) (= log(1/delta) + const) with a path bootstrap; first-order slope 1/D_P
            def fit(idx):
                xs = [np.log(1 / d) for d in DELTAS]; ys = [per[d][2][idx].mean() for d in DELTAS]
                return np.polyfit(xs, ys, 1)[0]
            b = [fit(rng.integers(0, NP, NP)) for _ in range(500)]
            slopes.append(dict(sigma=sig, hold=kind, slope=float(fit(np.arange(NP))), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), first_order=1 / DP))
    for r_ in rows: print({k: round(v, 4) if isinstance(v, float) else v for k, v in r_.items()})
    for s_ in slopes: print(s_)
    json.dump(dict(rows=rows, slopes=slopes), open('/home/nafisnaufal1426/aciids/astar/theory/sim_v6_match.json', 'w'), indent=1)
