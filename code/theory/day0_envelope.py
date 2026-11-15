"""Exact first-day envelope (Proposition: Neyman-Pearson for VaR). Calm s~N(0,1), shock s~N(0,sig^2), alpha=5%.
No lead: the side value at forecast tau is calm-distributed, so the best rule widens by a constant (convexified).
Lead: one side observation W ~ N(0, sig^2) under shock, N(0,1) under calm. The optimal rule sets the threshold pointwise:
q(w) = argmin_q P1(s>q) LR(w) + lam h0(q), with LR(w) = dG1/dG0(w). Compare with the frontier simulation's day-0 values."""
import numpy as np, pandas as pd
from scipy.stats import norm
a, sig = 0.05, 2.0; q0 = norm.ppf(1 - a)
def lbar(q): es = norm.pdf(q) - q * norm.sf(q); return (1 - a) * es + a * (q + es)
h0 = lambda q: lbar(q) - lbar(q0); p1 = lambda q: norm.sf(q / sig)
Q = np.linspace(q0 - 1, q0 + 8, 4001); H, P = h0(Q), p1(Q)
# no-lead envelope: lower convex hull of (H, P)
def hull_at(d):
    best = 1.0
    for i in range(0, len(Q), 8):
        for j in range(i, len(Q), 8):
            if H[i] <= d <= H[j] and H[j] > H[i]:
                t = (d - H[i]) / (H[j] - H[i]); best = min(best, (1 - t) * P[i] + t * P[j])
            elif abs(H[i] - d) < 1e-12: best = min(best, P[i])
    return best
w = np.linspace(-9, 9, 3601); dw = w[1] - w[0]; g0 = norm.pdf(w); g1 = norm.pdf(w / sig) / sig; LR = g1 / g0
rows = []
for lam in np.exp(np.linspace(-3, 9, 160)):
    # pointwise minimisation: minimise LR(w) P(q) + lam H(q) over the grid
    obj = LR[:, None] * P[None, :] + lam * H[None, :]; k = obj.argmin(1)
    v1 = np.sum(g1 * P[k]) * dw; c0 = np.sum(g0 * H[k]) * dw
    rows.append((c0, v1))
lead = pd.DataFrame(rows, columns=['delta', 'V0']).sort_values('delta')
fr = pd.read_csv('/home/nafisnaufal1426/aciids/astar/results/frontier.csv'); fr = fr[(fr.a == a) & (fr.sig == sig)]
for d in [0.0003, 0.001, 0.0024, 0.006]:
    nl = hull_at(d); ld = np.interp(d, lead.delta, lead.V0)
    sim_own = fr[fr.algo == 'own_cusum'].assign(dd=lambda x: (np.log(x.delta.clip(1e-9)) - np.log(d)).abs()).sort_values('dd').iloc[0]
    sim_lead = fr[(fr.algo == 'lead_aware') & (fr.K == 1)].assign(dd=lambda x: (np.log(x.delta.clip(1e-9)) - np.log(d)).abs()).sort_values('dd').iloc[0]
    print(f'delta={d}: exact day-0 optimum, no lead {nl:.3f} | with one lead observation {ld:.3f} || simulated own CUSUM d0 {sim_own.d0:.3f} (delta {sim_own.delta:.4f}), lead CUSUM d0 {sim_lead.d0:.3f} (delta {sim_lead.delta:.4f}) | Thm-3 bound p_eps(1-delta/kappa) best: ', round(max(p1(q0+e)*(1-d/h0(q0+e)) for e in np.linspace(0.01,3,300) if h0(q0+e)>d), 3))
