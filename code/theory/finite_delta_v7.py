"""v7 (round-5 R1/S22): exact per-day optimum and finite-delta per-shock lower bound for own feedback.

Setting: calm s ~ N(0,1), shock s ~ N(0, sigma^2), alpha = 5%, own feedback only (L <= -1, rule ignores X or X absent).
Per-day budget E_inf h0(q_t) <= delta for every t. On shock day d (d = 0, 1, ...), the information at forecast tau+d differs
from calm only through the d shocked own scores s_tau..s_{tau+d-1}; their likelihood ratio R_d = prod f1/f0 is a sufficient
statistic for the day-d problem, because the rest of the information has the same law under calm and shock and the
objective and constraint depend on the rule only through the conditional law of q given the information.
Proposition 1 (duality) applied with G0, G1 = laws of the information at forecast tau+d gives the smallest violation
probability on day d over rules with per-day calm cost delta:
    V_d(delta) = sup_{lam >= 0} E_{G1}[ min_q { P1(s > q) + lam h0(q) / R_d } ] - lam delta.
Any rule with per-day calm cost delta has P(e_{tau+d} = 1) >= V_d(delta) for every d, so E V_m >= sum_{d<m} V_d(delta)
(a finite-delta bound; it is the exact optimum for a rule told tau, and a lower bound for every rule).
Under G1, log R_d = 0.5 (sigma^2 - 1) chi2_d - d log sigma. Expectations use 4000 midpoint quantiles of chi2_d.
Writes finite_delta_v7.json."""
import json
import numpy as np
from scipy.stats import norm, chi2
A = 0.05; q0 = norm.ppf(1 - A)
El = lambda q: (1 - A) * (norm.pdf(q) - q * norm.sf(q)) + A * (q * norm.cdf(q) + norm.pdf(q))
h0 = lambda q: El(q) - El(q0)
QG = np.concatenate([np.linspace(-2.0, 2.4, 400), np.linspace(2.4, 30, 2000)[1:]])
H0 = h0(QG)


def V(d, delta, sig, nq=4000):
    P1 = norm.sf(QG / sig)
    if d == 0:
        logR = np.zeros(1); w = np.ones(1)
    else:
        u = (np.arange(nq) + 0.5) / nq; x = chi2.ppf(u, d); logR = 0.5 * (sig ** 2 - 1) * x - d * np.log(sig); w = np.full(nq, 1.0 / nq)
    invR = np.exp(-logR)
    def dual(lam):
        # E_G1 min_q [P1 + lam h0 / R]; rows: R values
        M = P1[None, :] + lam * invR[:, None] * H0[None, :]
        return float(w @ M.min(1)) - lam * delta
    lams = np.exp(np.linspace(np.log(1e-3), np.log(1e6), 120)); vals = [dual(l) for l in lams]
    i = int(np.argmax(vals)); lo, hi = lams[max(i - 1, 0)], lams[min(i + 1, len(lams) - 1)]
    for _ in range(40):  # golden-section refine on the concave dual
        a = lo + 0.382 * (hi - lo); b = lo + 0.618 * (hi - lo)
        if dual(a) > dual(b): hi = b
        else: lo = a
    return max(max(vals), dual(0.5 * (lo + hi)))


def job(t):
    sig, delta, d = t
    return t, V(d, delta, sig)


if __name__ == '__main__':
    from multiprocessing import Pool
    SIGS = [2.0, 3.0]; DELTAS = [1e-2, 2.4e-3, 1e-3, 1e-4]; DMAX = 60
    with Pool(200) as p:
        res = dict(p.map(job, [(s_, de, d) for s_ in SIGS for de in DELTAS for d in range(DMAX)]))
    out = []
    for sig in SIGS:
        DP = 0.5 * (sig ** 2 - 1 - np.log(sig ** 2)); p0 = float(norm.sf(q0 / sig))
        for delta in DELTAS:
            vd = np.array([res[(sig, delta, d)] for d in range(DMAX)]); cum = np.cumsum(vd)
            row = dict(sigma=sig, delta=delta, p0=p0, D_P=DP, V=[float(x) for x in vd], total_m10=float(cum[9]), total_m60=float(cum[-1]),
                       first_order=float(p0 * np.log(1 / delta) / DP))
            out.append(row)
            print(sig, delta, 'V0..5', np.round(vd[:6], 4), 'sum10', round(row['total_m10'], 3), 'sum60', round(row['total_m60'], 3), 'first-order', round(row['first_order'], 3), flush=True)
    json.dump(out, open('/home/nafisnaufal1426/aciids/astar/theory/finite_delta_v7.json', 'w'), indent=1)
