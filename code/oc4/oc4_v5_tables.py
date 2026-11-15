"""Supplement tables for v5: supp/sim_v5.tex (theory/sim_v4_slopes.json, sim_v5_slopes.json) and supp/v5_extra.tex (oc4/v5.json, ownstress.json)."""
import json
B = '/home/nafisnaufal1426/aciids/astar/'
s4 = json.load(open(B + 'theory/sim_v4_slopes.json')); s5 = json.load(open(B + 'theory/sim_v5_slopes.json'))
f = lambda x, d=2: f'{x:.{d}f}'
L = [r'\begin{tabular}{lrrrrl}\toprule', r' & $1/D_P$ & Known laws & Fixed-score & Ratio & \\\midrule']
for a, b in zip(s4, s5):
    lab = f"$\\sigma={a['sig']:g}$, $m={a['m']}$"
    L.append(f"{lab}, median delay & {f(a['inv_D'])} & {f(a['slope_known'])} ({f(a['slope_known_se'])}) & {f(a['slope_df'])} ({f(a['slope_df_se'])}) & {f(a['ratio'],1)} [{f(a['ratio_lo'],1)}, {f(a['ratio_hi'],1)}] & \\\\")
    L.append(f"\\quad mean delay & & {f(b['slope_known'])} ({f(b['slope_known_se'])}) & {f(b['slope_df'])} ({f(b['slope_df_se'])}) & {f(b['ratio'],1)} [{f(b['ratio_lo'],1)}, {f(b['ratio_hi'],1)}] & \\\\")
    L.append(f"\\quad excess ratio, equal $\\gamma$ (300/3k/30k) & & & & {f(a['excess_ratio_g300'])} / {f(a['excess_ratio_g3000'])} / {f(a['excess_ratio_g30000'])} & \\\\")
    mr = [x for x in b['matched_calm_cost'] if x['excess_ratio'] is not None]
    L.append(f"\\quad excess ratio, matched calm cost & & & & " + ' / '.join(f(x['excess_ratio']) for x in mr) + ' & \\\\')
    L.append(f"\\quad calm cost $\\times10^3$, $\\gamma=30\\ldots3\\cdot10^5$ & & " + '/'.join(f(1e3 * x, 2) for x in b['calm_cost_known']) + ' & ' + '/'.join(f(1e3 * x, 2) for x in b['calm_cost_df']) + ' & & \\\\')
    L.append(f"\\quad bursts detected (\\%) & & " + '/'.join(f(100 * x, 0) for x in b['detected_known']) + ' & ' + '/'.join(f(100 * x, 0) for x in b['detected_df']) + ' & & \\\\')
L.append(r'\bottomrule\end{tabular}')
open(B + 'manuscript/supp/sim_v5.tex', 'w').write('\n'.join(L) + '\n')
v = json.load(open(B + 'oc4/v5.json')); o = json.load(open(B + 'oc4/ownstress.json'))['results']
def ci(r, k='pct_of_tracker'): return f"{r[k]:.1f} [{r['lo']:.1f}, {r['hi']:.1f}]"
R = [r'\begin{tabular}{lrr}\toprule & $\alpha=1\%$ & $\alpha=5\%$\\\midrule', r'\multicolumn{3}{l}{\emph{Paired tick-loss differences, \% of the tracker}}\\']
for comp in ['buffer minus PID', 'buffer minus QR', 'buffer minus DtACI', 'buffer minus lag-1 inputs (all three)', 'US + others: real minus lag-1']:
    rr = [x for x in v['paired'] if x['comparison'] == comp]; R.append(f"{comp} & {ci(rr[0])} & {ci(rr[1])}\\\\")
for comp in ['own stress vs tracker', 'all four vs tracker', 'cross-market net of own stress (all four - own stress)', 'v4: cross-market net of own gap (all three - own gap)', 'all four - all three']:
    rr = [x for x in o if x['comparison'] == comp]; R.append(f"{comp.replace('own stress', 'own-stress')} & {ci(rr[0])} & {ci(rr[1])}\\\\")
R.append(r'\multicolumn{3}{l}{\emph{Stale opens: targets whose corr(open-to-close, prior US) interval includes 0}}\\')
ss = v['stale_subset']
for name in ['targets with corr interval including 0']:
    rr = [x for x in ss if x['set'] == name]; R.append(f"tick loss, nine targets & {rr[0]['tick_pct']:.1f} [{rr[0]['lo']:.1f}, {rr[0]['hi']:.1f}] & {rr[1]['tick_pct']:.1f} [{rr[1]['lo']:.1f}, {rr[1]['hi']:.1f}]\\\\")
rr = [x for x in ss if x['set'].startswith('US-led')]
R.append(f"US-led first day, nine targets ({rr[0]['onsets']} onsets) & {rr[0]['tracker']:.1f}$\\to${rr[0]['buffer']:.1f} ({rr[0]['diff']:.1f} [{rr[0]['lo']:.1f}, {rr[0]['hi']:.1f}]) & {rr[1]['tracker']:.1f}$\\to${rr[1]['buffer']:.1f} ({rr[1]['diff']:.1f} [{rr[1]['lo']:.1f}, {rr[1]['hi']:.1f}])\\\\")
R.append(r'\multicolumn{3}{l}{\emph{Canada, tick loss vs tracker (\%)}}\\')
for inp in ['US only', 'all three', 'own gap only']:
    rr = [x for x in v['canada'] if x['inputs'] == inp]; R.append(f"{inp} & {rr[0]['pct']:.1f} [{rr[0]['lo']:.1f}, {rr[0]['hi']:.1f}] & {rr[1]['pct']:.1f} [{rr[1]['lo']:.1f}, {rr[1]['hi']:.1f}]\\\\")
R.append(r'\multicolumn{3}{l}{\emph{First day, uplift-matched null (oc / cc timing)}}\\')
for tm in ['oc', 'cc']:
    rr = [x for x in v['uplift_null'] if x['timing'] == tm]
    R.append(f"{tm}: mean uplift; real change; null mean (min, max) & " + ' & '.join(f"{x['uplift_mean']:.2f}; {x['real_buffer_change']:.1f}; {x['nullC_mean_change']:.1f} ({x['nullC_min']:.1f}, {x['nullC_max']:.1f})" for x in rr) + '\\\\')
    R.append(f"{tm}: rates, real tracker$\\to$buffer; null tracker$\\to$null & " + ' & '.join(f"{x['real_tracker']:.1f}$\\to${x['real_buffer']:.1f}; {x['nullC_tracker_rate']:.1f}$\\to${x['nullC_rate']:.1f}" for x in rr) + '\\\\')
    R.append(f"{tm}: fixed uplift on the real flag, change (rate) & " + ' & '.join(f"{x['fixed_uplift_real_flag_change']:.1f} ({x['fixed_uplift_real_flag_rate']:.1f})" for x in rr) + '\\\\')
R.append(r'\multicolumn{3}{l}{\emph{VaR level and jumps (buffer / tracker)}}\\')
for tm in ['oc', 'cc']:
    rr = [x for x in v['jumps'] if x['timing'] == tm]
    R.append(f"{tm}: mean VaR ratio; mean $|\\Delta|$ (\\%); share of rises $>50\\%$ (\\%) & " + ' & '.join(f"{x['mean_var_ratio']:.3f}; {x['buffer_mean_abs_change']:.1f}/{x['tracker_mean_abs_change']:.1f}; {x['buffer_share_rise_gt50']:.2f}/{x['tracker_share_rise_gt50']:.2f}" for x in rr) + '\\\\')
R.append(r'\bottomrule\end{tabular}')
open(B + 'manuscript/supp/v5_extra.tex', 'w').write('\n'.join(R).replace('_', r'\_').replace(r'\_{', '_{') + '\n')
print('ok')
