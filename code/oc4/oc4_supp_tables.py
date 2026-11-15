"""Supplement tables for manuscript v4."""
import json, pickle
import pandas as pd
from pathlib import Path
A = json.load(open('analysis.json')); EV = json.load(open('event.json')); sel = pickle.load(open('selected.pkl', 'rb')); dev = pd.read_pickle('dev.pkl')
SIM = json.load(open('/home/nafisnaufal1426/aciids/astar/theory/sim_v4_slopes.json')); T = Path('/home/nafisnaufal1426/aciids/astar/manuscript/supp'); T.mkdir(exist_ok=True)
PC = 'prev.\\ close'; BS = chr(92)
NAME = {'Base': 'Base model', 'Tracker': 'Tracker', 'GCT-own': 'GCT, gap', 'GCT-us': 'GCT, US', 'GCT-glob': 'GCT, earlier sessions', 'GCT-comb': 'GCT, gap + earlier', 'QR': 'Quantile regression', 'PID': 'Conformal PID', 'DtACI': 'DtACI', 'LinBuf': 'Learned buffer'}
# selected settings and grid edges
rows = []
for (tm, a, m), cfg in sorted(sel.items(), key=lambda x: (x[0][0] != 'oc', x[0][1], list(NAME).index(x[0][2]))):
    x = dev[(dev.timing == tm) & (dev.alpha == a) & (dev.meth == m)]; g = [json.loads(c) for c in x.cfg]; edge = []
    for k in cfg:
        vals = sorted({c[k] for c in g})
        if len(vals) > 1 and cfg[k] in (vals[0], vals[-1]): edge.append(k.replace('_', r'\_'))
    ok = x[(x.viol - a).abs() <= 0.1 * a]
    SYM = {'eta': BS + 'eta', 'eta0': BS + 'eta_0', 'eta1': BS + 'eta_1', 'gamma': BS + 'gamma', 'r': 'r', 'eta_u': BS + 'eta_u', 'eta_w': BS + 'eta_w'}
    cs = ', '.join(f'${SYM[k]}={v:g}$' for k, v in cfg.items()) or '--'; edge = ['$' + SYM[e.replace(BS + '_', '_')] + '$' for e in edge]
    rows.append(f"{'open' if tm == 'oc' else PC} & {100 * a:g}\\% & {NAME[m]} & {len(x)} & {cs} & {', '.join(edge) or '--'} & {'yes' if len(ok) else 'no'}\\\\")
(T / 'selected.tex').write_text("\\begin{longtable}{lllrp{4.2cm}p{1.6cm}l}\\toprule Timing & $\\alpha$ & Method & Grid & Selected & At edge & Rule\\\\\\midrule\\endhead\n" + '\n'.join(rows) + "\n\\bottomrule\\end{longtable}\n")
# backtests
rows = []
for r in A['backtests']:
    yl = '--' if r['yellow'] is None else f"{r['yellow']:.1f}"; rd = '--' if r['red'] is None else f"{r['red']:.2f}"
    rows.append(f"{'open' if r['timing'] == 'oc' else PC} & {100 * r['alpha']:g}\\% & {NAME[r['method']]} & {int(r['kupiec_rej'])}/{r['n']} & {int(r['ind_rej'])}/{r['n']} & {yl} & {rd} & {r['viol_after_viol']:.1f}\\\\")
(T / 'backtests.tex').write_text("\\begin{longtable}{lllrrrrr}\\toprule Timing & $\\alpha$ & Method & Kupiec & Indep. & Yellow \\% & Red \\% & $e_{t+1}\\mid e_t$ \\%\\\\\\midrule\\endhead\n" + '\n'.join(rows) + "\n\\bottomrule\\end{longtable}\n")
# per target
rows = [f"{r['target']} & {r['vendor']} & {r['open_eq_high_or_low']:.1f} & {r['corr_oc_prior_us']:.3f} & {r['gain_0.01']:.1f} & {r['gain_0.05']:.1f}\\\\" for r in A['per_target']]
(T / 'per_target.tex').write_text("\\begin{tabular}{llrrrr}\\toprule Target & Vendor & Open $=$ high or low (\\%) & corr(open-to-close, prior US) & Gain 1\\% & Gain 5\\%\\\\\\midrule\n" + '\n'.join(rows) + "\n\\bottomrule\\end{tabular}\n")
# regions
rows = []
for rn in ['Asia', 'Europe', 'Canada', 'Southeast Asia']:
    a = [r for r in A['regions'] if r['region'] == rn and r['alpha'] == 0.01][0]; b = [r for r in A['regions'] if r['region'] == rn and r['alpha'] == 0.05][0]
    rows.append(f"{rn} & ${a['pct']:.1f}\\%$ [{a['lo']:.1f}, {a['hi']:.1f}] & ${b['pct']:.1f}\\%$ [{b['lo']:.1f}, {b['hi']:.1f}]\\\\")
(T / 'regions.tex').write_text("\\begin{tabular}{lll}\\toprule Region & $\\alpha=1\\%$ & $\\alpha=5\\%$\\\\\\midrule\n" + '\n'.join(rows) + "\n\\bottomrule\\end{tabular}\n")
# onset sensitivity
sg = pd.DataFrame(EV['sensitivity']); rows = []
for (Q, c, w), x in sg.groupby(['Q', 'c', 'w']):
    v = {(r.alpha, r.group): (r.diff, r.onsets) for r in x.itertuples()}
    f = lambda a, g: f"{v[(a, g)][0]:+.1f} ({v[(a, g)][1]})"
    rows.append(f"{Q} & {c:g} & $\\pm{w}$ & {f(0.01, 'US led')} & {f(0.01, 'Not US led')} & {f(0.05, 'US led')} & {f(0.05, 'Not US led')}\\\\")
(T / 'sensitivity.tex').write_text("\\begin{longtable}{rrrllll}\\toprule Quiet days & Share & Window & US led, 1\\% & Not led, 1\\% & US led, 5\\% & Not led, 5\\%\\\\\\midrule\\endhead\n" + '\n'.join(rows) + "\n\\bottomrule\\end{longtable}\n")
# simulation slopes
rows = []
for r in SIM:
    rows.append(f"$\\sigma={r['sig']:g}$, $m={r['m']}$ & {r['inv_D']:.2f} & {r['slope_known']:.2f} ({r['slope_known_se']:.2f}) & {r['slope_df']:.2f} ({r['slope_df_se']:.2f}) & {r['ratio']:.1f} [{r['ratio_lo']:.1f}, {r['ratio_hi']:.1f}] & "
                + ' / '.join(f"{r[f'excess_ratio_g{g}']:.2f}" for g in [300, 3000, 30000]) + "\\\\")
(T / 'sim_slopes.tex').write_text("\\begin{tabular}{lrrrll}\\toprule Burst & $1/D_P$ & Slope, known & Slope, fixed-score & Ratio & Excess ratio, $\\gamma=300$ / $3{,}000$ / $30{,}000$\\\\\\midrule\n" + '\n'.join(rows) + "\n\\bottomrule\\end{tabular}\n")
# GCT US placebo and event placebo A
g = [r for r in A['placebo'] if r['inputs'] == 'GCT on US stream']
ea = [r for tm in ['oc', 'cc'] for r in EV[tm] if r.get('label') == 'A shift-inputs' and r['group'] != 'all']
lines = [f"GCT on the US stream, US shifted (real-input steps): {100 * r['alpha']:g}\\%: real {r['real']:.3f}, shifted mean {r['mean']:.3f} (min {r['min']:.3f}), {r['n_le_real']}/{r['n']} draws at or below real." for r in g]
lines += [f"First-day null A (shifted inputs, real grouping), {'open' if r['timing'] == 'oc' else 'previous close'}, {100 * r['alpha']:g}\\%, {r['group']}: mean {r['null_mean']:+.1f} (min {r['null_min']:+.1f}), {r['n_le_real']}/{r['n']} at or below real." for r in ea]
(T / 'other_placebos.tex').write_text('\\begin{itemize}\n' + '\n'.join('\\item ' + l for l in lines) + '\n\\end{itemize}\n')
print('ok', [p.name for p in T.iterdir()])
