"""LaTeX tables for manuscript v4 from analysis.json, event.json, power.json and selected.pkl."""
import json, pickle
from pathlib import Path
A = json.load(open('/home/nafisnaufal1426/aciids/astar/oc4/analysis.json')); EV = json.load(open('/home/nafisnaufal1426/aciids/astar/oc4/event.json'))
T = Path('/home/nafisnaufal1426/aciids/astar/manuscript/tables')
def f2(x): return f'{x:.2f}'
def ci(d, lo, hi, nd=2):
    fm = lambda v: f'{v:.{nd}f}' if abs(v) >= 10 ** -nd / 2 else ('-0.00' if v < 0 else '0.00')
    return f'${d:+.{nd}f}$ [{fm(lo)}, {fm(hi)}]'
def get(rows, **kw): return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
NAME = {'Base': 'Base model alone', 'Tracker': 'Quantile tracker', 'GCT-own': 'GCT, own overnight gap', 'GCT-us': 'GCT, US session', 'GCT-glob': 'GCT, all earlier sessions',
        'GCT-comb': 'GCT, gap and earlier sessions', 'QR': 'Rolling quantile regression', 'PID': 'Conformal PID, QR scorecaster', 'DtACI': 'DtACI on QR residuals', 'LinBuf': 'Learned buffer'}
# ---- Table 1
L = [r"\begin{table}[t]\centering\caption{Exploratory study, 17 indices $\times$ 2 base models, 2015--2025. Upper block: open-to-close loss, VaR set at the local open. Lower block: close-to-close loss, VaR set at the previous local close. Every method sees only information time-stamped before the VaR is set; QR, PID, DtACI and the learned buffer see the same features. Viol.: violation rate (\%). Tick: tick loss $\times100$. $\Delta$: difference from the tracker, with a 95\% 21-day block-bootstrap interval.}\label{tab:emp}",
     r"\setlength{\tabcolsep}{2.0pt}\scriptsize\begin{tabular}{lrrlrrl}\toprule", r"& \multicolumn{3}{c}{$\alpha=1\%$} & \multicolumn{3}{c}{$\alpha=5\%$}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}", r"Method & Viol. & Tick & $\Delta$ & Viol. & Tick & $\Delta$\\\midrule"]
for tm, meths in [('oc', ['Base', 'Tracker', 'GCT-own', 'GCT-us', 'GCT-glob', 'GCT-comb', 'QR', 'PID', 'DtACI', 'LinBuf']), ('cc', ['Tracker', 'LinBuf'])]:
    if tm == 'cc': L.append(r"\midrule\multicolumn{7}{l}{\emph{VaR at the previous close}}\\")
    for m in meths:
        a, b = get(A[f'main_{tm}'], alpha=0.01, method=m)[0], get(A[f'main_{tm}'], alpha=0.05, method=m)[0]
        c1 = '--' if m == 'Tracker' else ci(a['d'], a['lo'], a['hi']); c2 = '--' if m == 'Tracker' else ci(b['d'], b['lo'], b['hi'])
        L.append(f"{NAME[m]} & {a['viol']:.2f} & {a['tick']:.2f} & {c1} & {b['viol']:.2f} & {b['tick']:.2f} & {c2}\\\\")
L.append(r"\bottomrule\end{tabular}\end{table}"); (T / 't_v4_main.tex').write_text('\n'.join(L) + '\n')
# ---- Table 2: inputs and placebos
L = [r"\begin{table}[t]\centering\caption{What drives the learned buffer (exploratory, open-to-close). Upper block: tick loss relative to the tracker (\%), 95\% interval. Lower block: tick loss $\times100$ when cross-market inputs are shifted by one common lag; long lags are 40 random draws of 250--750 days, shown as mean (minimum) / real value; number of draws at or below the real value. Step sizes are re-selected on 2007--2014 for every shifted input, except where marked. Tracker: 3.265 at 1\%, 10.279 at 5\%.}\label{tab:inputs}",
     r"\setlength{\tabcolsep}{2.0pt}\scriptsize\begin{tabular}{lll}\toprule", r"Inputs & $\alpha=1\%$ & $\alpha=5\%$\\\midrule"]
for mk, lab in [('own gap only', 'Own overnight gap only'), ('US only', 'US flag only'), ('other markets only', 'Other markets only'), ('US + others', 'US and other markets'), ('all three', 'All three')]:
    a, b = get(A['ablation'], alpha=0.01, inputs=mk)[0], get(A['ablation'], alpha=0.05, inputs=mk)[0]
    L.append(f"{lab} & ${a['pct']:.1f}\\%$ [{a['lo']:.1f}, {a['hi']:.1f}] & ${b['pct']:.1f}\\%$ [{b['lo']:.1f}, {b['hi']:.1f}]\\\\")
L.append(r"\midrule")
def prow(lab, **kw):
    a, b = get(A['placebo'], alpha=0.01, **kw)[0], get(A['placebo'], alpha=0.05, **kw)[0]
    fm = lambda r: f"{r['mean']:.2f} ({r['min']:.2f}) / {r['real']:.2f}; {r['n_le_real']}/{r['n']}"
    return f"{lab} & {fm(a)} & {fm(b)}\\\\"
L.append(prow('US flag shifted', family='US', inputs='US flag only', steps='re-tuned'))
L.append(prow('\\quad same, real-input steps', family='US', inputs='US flag only', steps='real-tuned'))
L.append(prow('US and others shifted', family='X', inputs='US + others', steps='re-tuned'))
L.append(prow('All three, cross-market shifted', family='X', inputs='all three', steps='re-tuned'))
a, b = get(A['placebo'], alpha=0.01, family='X', inputs='all three', steps='re-tuned')[0], get(A['placebo'], alpha=0.05, family='X', inputs='all three', steps='re-tuned')[0]
L.append(f"\\quad all three lagged 1 / 2 / 3 / 5 days & {a['lag1']:.2f} / {a['lag2']:.2f} / {a['lag3']:.2f} / {a['lag5']:.2f} & {b['lag1']:.2f} / {b['lag2']:.2f} / {b['lag3']:.2f} / {b['lag5']:.2f}\\\\")
a, b = get(A['placebo'], alpha=0.01, family='X', inputs='US + others', steps='re-tuned')[0], get(A['placebo'], alpha=0.05, family='X', inputs='US + others', steps='re-tuned')[0]
L.append(f"\\quad US and others lagged 1 / 2 / 3 / 5 days & {a['lag1']:.2f} / {a['lag2']:.2f} / {a['lag3']:.2f} / {a['lag5']:.2f} & {b['lag1']:.2f} / {b['lag2']:.2f} / {b['lag3']:.2f} / {b['lag5']:.2f}\\\\")
L.append(prow('Previous close, cross-market shifted', family='CC', steps='re-tuned'))
L.append(r"\bottomrule\end{tabular}\end{table}"); (T / 't_v4_inputs.tex').write_text('\n'.join(L) + '\n')
# ---- Table 3: first day
L = [r"\begin{table}[t]\centering\caption{First-day violation rate (\%) at shock onsets, 2015--2025 (exploratory). Rates pool both base models ($2n$ records). Tr.: tracker. Buf.: learned buffer. $\Delta$: Buf.\ minus Tr., with a 95\% wild cluster bootstrap interval over episodes (Webb weights). Null: mean $\Delta$ of 40 draws in which the buffer sees shifted inputs and the shifted US flag defines the group. US led: the US flag the buffer sees was on. $n$: onsets (episodes).}\label{tab:event}",
     r"\setlength{\tabcolsep}{1.6pt}\scriptsize\begin{tabular}{lrrrlrrrlr}\toprule", r"& & \multicolumn{4}{c}{$\alpha=1\%$} & \multicolumn{4}{c}{$\alpha=5\%$}\\\cmidrule(lr){3-6}\cmidrule(lr){7-10}", r"Group & $n$ & Tr. & Buf. & $\Delta$ & Null & Tr. & Buf. & $\Delta$ & Null\\\midrule"]
for tm, head in [('oc', r'\emph{VaR at the open}'), ('cc', r'\emph{VaR at the previous close}')]:
    L.append(f"\\multicolumn{{10}}{{l}}{{{head}}}\\\\")
    for g, lab in [('US led', 'US led'), ('Not US led', 'Not US led')]:
        cells = []
        for a in [0.01, 0.05]:
            r = get(EV[tm], alpha=a, label='real', group=g)[0]; nb = get(EV[tm], alpha=a, label='B self-grouped', group=g)[0]
            cells.append(f"{r['tracker']:.1f} & {r['buffer']:.1f} & ${r['diff']:+.1f}$ [{r['wild_lo']:.1f}, {r['wild_hi']:.1f}] & ${nb['null_mean']:+.1f}$")
        r = get(EV[tm], alpha=0.01, label='real', group=g)[0]
        L.append(f"{lab} & {r['onsets']:.0f} ({r['episodes']}) & {cells[0]} & {cells[1]}\\\\")
L.append(r"\bottomrule\end{tabular}\end{table}"); (T / 't_v4_event.tex').write_text('\n'.join(L) + '\n')
# ---- Table 4: all days by US flag
L = [r"\begin{table}[t]\centering\caption{All test days by the US flag the buffer sees (open-to-close, exploratory). Local exceedance: the own loss exceeds its rolling 95\% quantile. Share: \% of target-days. Viol.: violation rate (\%), tracker / buffer. VaR: buffer VaR over tracker VaR. Gain: share of the buffer's net tick-loss gain earned on these days (negative: a cost).}\label{tab:flags}",
     r"\setlength{\tabcolsep}{2.2pt}\scriptsize\begin{tabular}{lrrrrrrr}\toprule", r"& & \multicolumn{3}{c}{$\alpha=1\%$} & \multicolumn{3}{c}{$\alpha=5\%$}\\\cmidrule(lr){3-5}\cmidrule(lr){6-8}", r"Days & Share & Viol. & VaR & Gain & Viol. & VaR & Gain\\\midrule"]
for g, lab in [('flag on, local exceedance', 'Flag on, local exceedance'), ('flag on, no local exceedance (false alarm)', 'Flag on, none (false alarm)'), ('flag off, local exceedance', 'Flag off, local exceedance'), ('flag off, no local exceedance', 'Flag off, none')]:
    a, b = get(A['flag_days'], alpha=0.01, group=g)[0], get(A['flag_days'], alpha=0.05, group=g)[0]
    L.append(f"{lab} & {a['share']:.1f} & {a['viol_tracker']:.1f} / {a['viol_buffer']:.1f} & {a['var_ratio']:.2f} & {100 * a['gain_share']:.0f}\\% & {b['viol_tracker']:.1f} / {b['viol_buffer']:.1f} & {b['var_ratio']:.2f} & {100 * b['gain_share']:.0f}\\%\\\\")
L.append(r"\bottomrule\end{tabular}\end{table}"); (T / 't_v4_flags.tex').write_text('\n'.join(L) + '\n')
print('\n\n'.join((T / f).read_text() for f in ['t_v4_main.tex', 't_v4_inputs.tex', 't_v4_event.tex', 't_v4_flags.tex']))
