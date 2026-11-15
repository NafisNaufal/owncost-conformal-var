"""v5 EXPLORATORY (round-3 R22): separate cross-market information from the target's own previous-session stress.
Adds a fourth buffer feature, the target's own previous close-to-close exceedance (loss above its rolling 250-day 95% quantile
of prior data, the last close before the open; no look-ahead), and re-tunes the buffer step sizes on 2007-2014 with the same
rule as oc4_study.py for two feature sets:
  own stress  - own squared gap and own previous exceedance (no cross-market input);
  all four    - own squared gap, US flag, share of earlier sessions, own previous exceedance.
Reports test-window (2015-2025) tick loss against the tracker, and the cross-market increment net of own stress
(all four minus own stress) beside the v4 increment (all three minus own gap only). Writes ownstress.json and ownstress_cells.pkl."""
import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/oc4')
import json, pickle
from multiprocessing import Pool
import numpy as np, pandas as pd
import oc4_lib as L
import oc4_study as S
import oc4_eval as E
D = S.D; sel = pickle.load(open('/home/nafisnaufal1426/aciids/astar/oc4/selected.pkl', 'rb'))
DATA = {m: L.load_ohlc(m) for m in S.TG}
OWNPREV = {}
for k in S.TG:
    l = -DATA[k].r; thr = l.shift(1).rolling(250).quantile(0.95); ex = ((l > thr) & thr.notna()).astype(float)
    prev = ex.shift(1)  # exceedance of the previous trading day's close-to-close loss, known at today's open
    OWNPREV[k] = prev.reindex(pd.DatetimeIndex(D[k]['oc']['dates'])).fillna(0.0).to_numpy()
FS = {'own stress': (1, 0, 0, 1), 'all four': (1, 1, 1, 1)}


def F4(k, mask):
    F = np.column_stack([D[k]['oc']['F'], OWNPREV[k]]); return F * np.asarray(mask, float)[None, :]


def run4(k, b, a, cfg, mask):
    s, v, ok = S.scores(k, b, a, 'oc'); i0 = int(np.argmax(ok)); ss = s[i0:]
    q = L.linear_buffer(ss, F4(k, mask)[i0:], a, cfg['eta_u'], cfg['eta_w'], 6.0, S.BURN)
    Q = np.full(len(s), np.nan); Q[i0:] = q; vv = v + D[k]['oc']['sig'] * Q; Lk = D[k]['oc']['loss']
    e = np.where(np.isfinite(vv), (Lk > vv).astype(float), np.nan); u = vv - Lk; tk = (a - (u < 0)) * u
    return e, tk, vv


def dev_job(t):
    name, a, ci = t; cfg = E.BUFGRID[ci]; tks, vs = [], []
    for k in S.TG:
        for b in S.BASES:
            e, tk, vv = run4(k, b, a, cfg, FS[name]); x, y = S.score_window(k, e, tk, *L.DEV, 'oc'); tks.append(x); vs.append(y)
    return (name, a, ci), (float(np.mean(tks)), float(np.mean(vs)))


def test_job(t):
    name, k, b, a, cfg = t; e, tk, vv = run4(k, b, a, cfg, FS[name]); return (name, k, b, a), dict(e=e.astype(np.float32), tk=tk.astype(np.float32), vv=vv.astype(np.float32))


if __name__ == '__main__':
    tasks = [(n, a, ci) for n in FS for a in S.LV for ci in range(len(E.BUFGRID))]
    with Pool(64) as p:
        dres = dict(p.map(dev_job, tasks))
    chosen = {}
    for n in FS:
        for a in S.LV:
            rows = [(ci,) + dres[(n, a, ci)] for ci in range(len(E.BUFGRID))]; ok = [r for r in rows if abs(r[2] - a) <= 0.1 * a]
            best = min(ok, key=lambda r: r[1]) if ok else min(rows, key=lambda r: abs(r[2] - a)); chosen[(n, a)] = E.BUFGRID[best[0]]
            print('dev', n, a, E.BUFGRID[best[0]], 'tick', round(100 * best[1], 3), 'viol', round(100 * best[2], 2), 'rule_met', bool(ok), flush=True)
    T = [(n, k, b, a, chosen[(n, a)]) for n in FS for k in S.TG for b in S.BASES for a in S.LV]
    with Pool(64) as p:
        RC = dict(p.map(test_job, T))
    pickle.dump(dict(cells=RC, chosen={f'{n}|{a}': c for (n, a), c in chosen.items()}), open('/home/nafisnaufal1426/aciids/astar/oc4/ownstress_cells.pkl', 'wb'))
    # analysis on the common date grid with the same block bootstrap as oc4_analysis.py
    Rv4 = pickle.load(open('/home/nafisnaufal1426/aciids/astar/oc4/cells.pkl', 'rb'))
    lo, hi = L.TEST; U = np.unique(np.concatenate([D[k]['oc']['dates'] for k in S.TG])); M = (U >= np.datetime64(lo)) & (U < np.datetime64(hi))
    def og(k, x): out = np.full(len(U), np.nan); out[np.searchsorted(U, D[k]['oc']['dates'])] = x; return out[M]
    def st(fn): return np.column_stack([og(k, fn(k, b)) for k in S.TG for b in S.BASES])
    rng = np.random.default_rng(12); Tn = int(M.sum()); stt = rng.integers(0, Tn, (3000, int(np.ceil(Tn / 21)))); I = ((stt[:, :, None] + np.arange(21)).reshape(3000, -1)[:, :Tn]) % Tn
    def ci(Mx):
        s = np.nansum(Mx, 1); c = np.sum(np.isfinite(Mx), 1).astype(float); b = s[I].sum(1) / c[I].sum(1); return s.sum() / c.sum(), np.percentile(b, 2.5), np.percentile(b, 97.5)
    out = []
    for a in S.LV:
        Tt = st(lambda k, b: Rv4[('oc', k, b, a, 'Tracker')]['tk']); mt = np.nanmean(Tt)
        own = st(lambda k, b: RC[('own stress', k, b, a)]['tk']); all4 = st(lambda k, b: RC[('all four', k, b, a)]['tk'])
        gap = st(lambda k, b: Rv4[('abl', k, b, a, 'own gap only')]['tk']); all3 = st(lambda k, b: Rv4[('abl', k, b, a, 'all three')]['tk'])
        for name, X, Y in [('own stress vs tracker', own, Tt), ('all four vs tracker', all4, Tt), ('cross-market net of own stress (all four - own stress)', all4, own),
                           ('v4: cross-market net of own gap (all three - own gap)', all3, gap), ('all four - all three', all4, all3)]:
            e_, l_, h_ = ci(X - Y); out.append(dict(alpha=a, comparison=name, pct_of_tracker=100 * e_ / mt, lo=100 * l_ / mt, hi=100 * h_ / mt))
        for name in FS:
            out.append(dict(alpha=a, comparison=f'violation rate, {name}', pct_of_tracker=100 * float(np.nanmean(st(lambda k, b, n=name: RC[(n, k, b, a)]['e'])))))
    print(pd.DataFrame(out).round(2).to_string(index=False))
    json.dump(dict(results=out, chosen={f'{n}|{a}': c for (n, a), c in chosen.items()}), open('/home/nafisnaufal1426/aciids/astar/oc4/ownstress.json', 'w'), indent=1, default=float)
