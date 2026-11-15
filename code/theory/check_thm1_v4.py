"""Numerical check of the restated Theorem 1 (v4):
P(e_{tau+d}=1) >= p_eps [1 - exp(K_d + x) delta/kappa_eps - W_d/x^2], K_d = n_X(d) D_X + n_s(d) D_P, n_X = (d+L+1)_+, n_s = d
(n_s = 0 for rules that ignore own scores). Checks it on (1) the round-2 own-score counterexample, (2) the L = 0 two-point
first-day counterexample, (3) own-feedback rules, by Monte Carlo against the bound maximised over eps and x."""
import numpy as np
from scipy.stats import norm, chi2
from scipy.optimize import minimize_scalar
a = 0.05; q0 = norm.ppf(1 - a); sd1 = 2.0
def h0(q):  # calm cost of threshold q under N(0,1): E[l(q,s)-l(q0,s)]
    f = lambda q: (1 - a) * (norm.pdf(q) - q * norm.sf(q)) + a * (q - (-(norm.pdf(q)) + q * norm.cdf(q)) * 0 - (q * norm.cdf(q) + norm.pdf(q)) + q * 0) if False else None
    # closed form: E l(q,s) = (1-a) E(s-q)_+ + a E(q-s)_+ ; E(s-q)_+ = phi(q) - q Sbar(q); E(q-s)_+ = q Phi(q) + phi(q)
    El = lambda q: (1 - a) * (norm.pdf(q) - q * norm.sf(q)) + a * (q * norm.cdf(q) + norm.pdf(q))
    return El(q) - El(q0)
KL = lambda s1: 0.5 * (s1 ** 2 - 1 - np.log(s1 ** 2))          # KL(N(0,s1^2) || N(0,1))
def var_llr(s1, n=400000, rng=np.random.default_rng(0)):
    x = rng.normal(0, s1, n); l = -np.log(s1) - x ** 2 / (2 * s1 ** 2) + x ** 2 / 2; return l.var()
DP = KL(sd1); VP = var_llr(sd1)
def bound(delta, K, W):
    best = 0.0
    for eps in np.linspace(0.01, 3, 300):
        kap = h0(q0 + eps); pe = norm.sf((q0 + eps) / sd1)
        if K == 0 and W == 0:
            best = max(best, pe * (1 - delta / kap)); continue
        r = minimize_scalar(lambda x: -(1 - np.exp(K + x) * delta / kap - W / x ** 2), bounds=(1e-3, 60), method='bounded')
        best = max(best, pe * max(0, -r.fun))
    return best
rng = np.random.default_rng(1)
# (1) own-score counterexample: rule ignores X, q=4 if sum of last 10 squared own scores > chi2_10 upper 2%; side X with D_X=5e-7, L=0
c = chi2.ppf(0.98, 10); delta1 = 0.02 * h0(4.0); d = 10
viol = 0.02 * norm.sf(4 / sd1) + 0  # under shock the last 10 scores are N(0,4): P(sum s^2 > c) with s ~ N(0,4) => P(chi2_10 > c/4)
pr = chi2.sf(c / 4, 10); viol = pr * norm.sf(4 / sd1) + (1 - pr) * norm.sf(q0 / sd1)
DX, VX = 5e-7, 1e-6
b_old = bound(delta1, (d + 0) * DX, (d + 0) * VX); b_new = bound(delta1, (d + 1) * DX + d * DP, (d + 1) * VX + d * VP)
print(f'(1) own-score rule: calm cost {delta1:.5f}, P(viol at d=10) {viol:.4f}; v3 bound {b_old:.4f} (violated: {viol < b_old}); v4 bound {b_new:.4f} (holds: {viol >= b_new - 1e-9})')
# (2) L = 0 first day, informative side flag: X_tau ~ Bern(0.9) in shock vs Bern(0.002) in calm; rule q = 5 if X=1 else q0
p1, p0x = 0.9, 0.002; delta2 = p0x * h0(5.0)
viol2 = p1 * norm.sf(5 / sd1) + (1 - p1) * norm.sf(q0 / sd1)
DX2 = p1 * np.log(p1 / p0x) + (1 - p1) * np.log((1 - p1) / (1 - p0x)); llr = np.array([np.log(p1 / p0x), np.log((1 - p1) / (1 - p0x))]); VX2 = p1 * llr[0] ** 2 + (1 - p1) * llr[1] ** 2 - DX2 ** 2
print(f'(2) L=0 first day: calm cost {delta2:.5f}, P(viol) {viol2:.4f}; v3 bound (count 0) {bound(delta2, 0, 0):.4f} (violated: {viol2 < bound(delta2, 0, 0)}); v4 bound (count 1) {bound(delta2, DX2, VX2):.4f}')
# (3) own-feedback tracker, Monte Carlo: violations at d=0..5 vs v4 bound with n_s = d
eta = 0.05; M = 20000; T0 = 3000
u = np.full(M, q0)
for t in range(T0):
    s = rng.normal(0, 1, M); u += eta * ((s > u) - a)
calm_cost = h0(u).mean(); ok = True
for dd in range(6):
    s = rng.normal(0, sd1, M); v = (s > u).mean(); b = bound(calm_cost, dd * DP, dd * VP); ok &= v >= b - 3 * np.sqrt(v * (1 - v) / M)
    print(f'(3) tracker eta={eta}: d={dd} viol {v:.4f} >= v4 bound {b:.4f}'); u += eta * ((s > u) - a)
print('own-feedback check passes:', ok)
