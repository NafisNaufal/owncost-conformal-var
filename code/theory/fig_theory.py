"""Figure 1 for manuscript v6 (panel b: Theorem 3 check from sim_v6_match.json): (a) exact first-day optimum vs calm cost, (b) detection delay, oracle vs distribution-free."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import norm
TRK, OWN, CMQ, INK, MUT = '#D55E00', '#0072B2', '#009E73', '#1a1a19', '#6b6b67'
plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.5, 'lines.linewidth': 1.1, 'legend.frameon': False, 'pdf.fonttype': 42})
a, sig = 0.05, 2.0; q0 = norm.ppf(1 - a)
def lbar(q): es = norm.pdf(q) - q * norm.sf(q); return (1 - a) * es + a * (q + es)
h0 = lambda q: lbar(q) - lbar(q0); p1 = lambda q: norm.sf(q / sig)
Q = np.linspace(q0, q0 + 6, 3000); H, P = h0(Q), p1(Q)
# lower convex envelope of (H, P) via hull on sorted H
pts = sorted(zip(H, P)); hull = []
for x, y in pts:
    while len(hull) >= 2 and (hull[-1][1] - hull[-2][1]) * (x - hull[-2][0]) >= (y - hull[-2][1]) * (hull[-1][0] - hull[-2][0]): hull.pop()
    hull.append((x, y))
hx, hy = map(np.array, zip(*hull))
w = np.linspace(-9, 9, 3601); dw = w[1] - w[0]; g0 = norm.pdf(w); g1 = norm.pdf(w / sig) / sig; LR = g1 / g0; rows = []
Qs, Hs, Ps = Q[::3], H[::3], P[::3]
for lam in np.exp(np.linspace(-3, 9, 200)):
    k = (LR[:, None] * Ps[None, :] + lam * Hs[None, :]).argmin(1); rows.append((np.sum(g0 * Hs[k]) * dw, np.sum(g1 * Ps[k]) * dw))
ld = pd.DataFrame(rows, columns=['d', 'v']).sort_values('d')
fr = pd.read_csv('results/frontier.csv'); fr = fr[(fr.a == a) & (fr.sig == sig) & (fr.delta > 1e-4)]
dd = np.logspace(-4, np.log10(0.012), 60)
bound = [max([p1(q0 + e) * (1 - d / h0(q0 + e)) for e in np.linspace(0.01, 3, 300) if h0(q0 + e) > d] + [0]) for d in dd]
fig, ax = plt.subplots(1, 2, figsize=(4.8, 2.25))
ax[0].plot(dd, np.interp(dd, hx, hy), color=OWN, label='Exact optimum, $L\\leq-1$ (Prop. 1)')
ax[0].plot(dd, np.interp(dd, ld.d, ld.v), color=CMQ, label='Exact optimum, $L=0$ (Prop. 1)')
ax[0].plot(dd, bound, color=INK, ls=':', label='Theorem 1 bound, $L\\leq-1$')
for algo, K, col, mk, lab in [('tracker', 0, TRK, 'o', 'Tracker (sim.)'), ('own_cusum', 0, OWN, 's', 'Own-score CUSUM (sim.)'), ('lead_aware', 1, CMQ, '^', 'Side CUSUM, $L=0$ (sim.)')]:
    x = fr[(fr.algo == algo) & (fr.K == K)]; ax[0].scatter(x.delta, x.d0, s=9, color=col, marker=mk, lw=0, label=lab)
ax[0].set_xscale('log'); ax[0].set_xlabel('Calm cost $\\delta$'); ax[0].set_ylabel('First-day violation prob.'); ax[0].set_title('(a) First day, $\\alpha$=5%, burst N(0,4)', fontsize=7, loc='left')
S = json.load(open('theory/sim_v6_match.json'))['rows']
for sg, col in [(2.0, INK), (3.0, '#CC79A7')]:
    for hold, ls, mk, lab in [('conservative', '-', 'o', 'conservative hold'), ('nominal', '--', 's', 'nominal hold')]:
        x = sorted([r for r in S if r['sigma'] == sg and r['hold'] == hold], key=lambda r: -r['delta'])
        ax[1].plot([np.log(1 / r['delta']) for r in x], [r['V'] for r in x], ls=ls, marker=mk, ms=3, color=col, label=f"$\\sigma$={sg:g}, {lab}")
    x = sorted([r for r in S if r['sigma'] == sg and r['hold'] == 'conservative'], key=lambda r: -r['delta'])
    ax[1].plot([np.log(1 / r['delta']) for r in x], [r['first_order_V'] for r in x], ls=':', color=col, label=f"$\\sigma$={sg:g}, first order")
ax[1].set_xlabel('$\\log(1/\\delta)$'); ax[1].set_ylabel('Violations per burst'); ax[1].set_title('(b) Theorem 3, known laws, $m$=60', fontsize=7, loc='left')
for a_ in ax: a_.grid(axis='y', lw=0.3, color='#e4e4e0')
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=5.2, columnspacing=0.6, handlelength=1.6); ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=5.2, columnspacing=0.6, handlelength=1.6)
fig.tight_layout(w_pad=1.2); fig.savefig('figures/fig_v6_theory.pdf', bbox_inches='tight'); fig.savefig('figures/fig_v6_theory.png', dpi=220, bbox_inches='tight'); print('ok')
