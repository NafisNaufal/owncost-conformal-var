"""Numbers for manuscript v5 (round-3 R15, R21, S27). Own-feedback case, alpha = 5%, calm N(0,1), shock N(0, sigma^2).
(1) Theorem 1's finite-delta display at the paper's operating point delta = 0.0024, summed over shock days, with the Chebyshev
    term of the theorem and with the exact tail of the log-likelihood ratio (a scaled chi-square), against the first-order scale.
(2) The calm cost below which Theorem 2's fixed-step lower bound exceeds the first-order own-feedback price p0 log(1/delta)/D_P."""
import json
import numpy as np
from scipy.stats import norm, chi2
from scipy.optimize import brentq
a = 0.05; q0 = norm.ppf(1 - a); C0 = 2.507
El = lambda q: (1 - a) * (norm.pdf(q) - q * norm.sf(q)) + a * (q * norm.cdf(q) + norm.pdf(q))
h0 = lambda q: El(q) - El(q0)
EPS = np.linspace(0.01, 3, 300)
out = {}
for sig in [2.0, 3.0]:
    DP = 0.5 * (sig ** 2 - 1 - np.log(sig ** 2)); p0 = norm.sf(q0 / sig)
    VP = ((sig ** 2 - 1) / 2) ** 2 * 2  # variance of (sig^2-1)/2 * chi2_1
    def cheb(delta, d):
        K, W = d * DP, d * VP; best = 0.0
        for e in EPS:
            kap = h0(q0 + e); pe = norm.sf((q0 + e) / sig)
            if d == 0: best = max(best, pe * (1 - delta / kap)); continue
            xs = np.linspace(0.05, 60, 2400); val = 1 - np.exp(K + xs) * delta / kap - W / xs ** 2; best = max(best, pe * max(0, val.max()))
        return best
    def exact(delta, d):
        best = 0.0
        for e in EPS:
            kap = h0(q0 + e); pe = norm.sf((q0 + e) / sig)
            if d == 0: best = max(best, pe * (1 - delta / kap)); continue
            cs = np.linspace(-d * np.log(sig) + 1e-6, d * DP + 40, 3000)
            tail = chi2.sf((cs + d * np.log(sig)) * 2 / (sig ** 2 - 1), d)
            val = 1 - np.exp(cs) * delta / kap - tail; best = max(best, pe * max(0, val.max()))
        return best
    delta = 0.0024; ch = [cheb(delta, d) for d in range(40)]; ex = [exact(delta, d) for d in range(40)]
    first = p0 * np.log(1 / delta) / DP
    # (2) crossover: Theorem 2 (fixed-step) lower bound max_eps p_eps * eps*alpha/(4 C0 delta) vs p0 log(1/delta)/D_P
    def thm_fixed(dl): return max(norm.sf((q0 + e) / sig) * e * a / (4 * C0 * dl) for e in EPS)
    g = lambda ld: thm_fixed(np.exp(-ld)) - p0 * ld / DP
    lds = np.linspace(1, 30, 600); vals = [g(x) for x in lds]; idx = next(i for i, v in enumerate(vals) if v > 0)
    cross = float(np.exp(-brentq(g, lds[idx - 1], lds[idx])))
    out[f'sigma={sig:g}'] = dict(D_P=DP, p0=p0, first_order_price=first, days_first_order=float(np.log(1 / delta) / DP),
                                 cheb_by_day=ch[:4], cheb_total=float(sum(ch)), exact_by_day=ex[:4], exact_total=float(sum(ex)),
                                 fixed_step_bound_at_0024=thm_fixed(delta), crossover_delta=cross)
for k, v in out.items(): print(k, {kk: (np.round(vv, 4).tolist() if isinstance(vv, list) else round(float(vv), 5)) for kk, vv in v.items()})
json.dump(out, open('/home/nafisnaufal1426/aciids/astar/theory/v5_numbers.json', 'w'), indent=1, default=float)
