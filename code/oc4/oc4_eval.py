"""v4 EXPLORATORY evaluation cells (2015-2025 test window; dev-retuned placebos). Writes cells.pkl; analysis in oc4_analysis.py.
Placebo families (cross-market inputs shifted by a common lag; own gap kept real):
  US   - only the US flag (and, for GCT-us, the US stream) shifted, 40 long lags (250-750 days);
  X    - US flag and share shifted, inputs 'all three' and 'US + others', 40 long lags plus short lags 1, 2, 3, 5 days;
  CC   - previous-close timing, US flag and share shifted, 40 long lags.
For every LinBuf placebo the two step sizes are re-selected on 2007-2014 with the shifted inputs (same rule as the real run)."""
import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/oc4')
import itertools, json, pickle
from multiprocessing import Pool
import numpy as np, pandas as pd
import oc4_lib as L
import oc4_study as S
D = S.D; sel = pickle.load(open('/home/nafisnaufal1426/aciids/astar/oc4/selected.pkl', 'rb'))
OCM = ['Base', 'Tracker', 'GCT-own', 'GCT-us', 'GCT-glob', 'GCT-comb', 'QR', 'PID', 'DtACI', 'LinBuf']; CCM = ['Base', 'Tracker', 'GCT-us', 'LinBuf']
LONG = [int(x) for x in np.random.default_rng(7).integers(250, 751, 40)]; SHORT = [1, 2, 3, 5]
MASKS = {'own gap only': (1, 0, 0), 'US only': (0, 1, 0), 'other markets only': (0, 0, 1), 'US + others': (0, 1, 1), 'all three': (1, 1, 1)}
BUFGRID = [dict(eta_u=u, eta_w=w) for u, w in itertools.product([0.02, 0.05, 0.1, 0.2], [0.1, 0.5, 2.0, 5.0])]


def pack(k, tm, e, tk, st, vv):
    return dict(e=e.astype(np.float32), tk=tk.astype(np.float32), st=st.astype(np.int8), vv=vv.astype(np.float32))


def cell(t):
    key, (k, b, a, m, cfg, tm, shift, fmask, cols) = t
    e, tk, st, vv = S.run(k, b, a, m, cfg, tm, shift, 0, fmask, cols)
    return key, pack(k, tm, e, tk, st, vv)


def dev_placebo(t):
    fam, mask, lag, a, ci = t; cfg = BUFGRID[ci]; tm = 'cc' if fam == 'CC' else 'oc'
    cols = [1] if fam == 'US' else None; tks, vs = [], []
    for k in S.TG:
        for b in S.BASES:
            e, tk, st, vv = S.run(k, b, a, 'LinBuf', cfg, tm, lag, 0, mask, cols); x, y = S.score_window(k, e, tk, *L.DEV, tm); tks.append(x); vs.append(y)
    return (fam, mask, lag, a, ci), (float(np.mean(tks)), float(np.mean(vs)))


def choose(res, fam, mask, lag, a):
    rows = [(ci,) + res[(fam, mask, lag, a, ci)] for ci in range(len(BUFGRID))]
    ok = [r for r in rows if abs(r[2] - a) <= 0.1 * a]
    best = min(ok, key=lambda r: r[1]) if ok else min(rows, key=lambda r: abs(r[2] - a))
    return BUFGRID[best[0]]


if __name__ == '__main__':
    fams = [('US', None, l) for l in LONG] + [('X', MASKS[m], l) for m in ['all three', 'US + others'] for l in LONG + SHORT] + [('CC', None, l) for l in LONG]
    dtasks = [(f, m, l, a, ci) for f, m, l in fams for a in S.LV for ci in range(len(BUFGRID))]
    print('placebo dev jobs', len(dtasks), flush=True)
    with Pool(220) as p:
        dres = dict(p.map(dev_placebo, dtasks, chunksize=8))
    psel = {(f, m, l, a): choose(dres, f, m, l, a) for f, m, l in fams for a in S.LV}
    pickle.dump(psel, open('/home/nafisnaufal1426/aciids/astar/oc4/placebo_selected.pkl', 'wb'))
    T = []
    for k in S.TG:
        for b in S.BASES:
            for a in S.LV:
                T += [(('oc', k, b, a, m), (k, b, a, m, sel[('oc', a, m)], 'oc', 0, None, None)) for m in OCM]
                T += [(('cc', k, b, a, m), (k, b, a, m, sel[('cc', a, m)], 'cc', 0, None, None)) for m in CCM]
                T += [(('abl', k, b, a, mk), (k, b, a, 'LinBuf', sel[('oc', a, 'LinBuf')], 'oc', 0, MASKS[mk], None)) for mk in MASKS]
                for f, m, l in fams:
                    tm = 'cc' if f == 'CC' else 'oc'; cols = [1] if f == 'US' else None
                    T.append((('pl', f, m, l, k, b, a), (k, b, a, 'LinBuf', psel[(f, m, l, a)], tm, l, m, cols)))
                    T.append((('plfix', f, m, l, k, b, a), (k, b, a, 'LinBuf', sel[(tm, a, 'LinBuf')], tm, l, m, cols)))
                T += [(('plgct', l, k, b, a), (k, b, a, 'GCT-us', sel[('oc', a, 'GCT-us')], 'oc', l, None, [1])) for l in LONG]
    print('cells', len(T), flush=True)
    with Pool(220) as p:
        R = dict(p.map(cell, T, chunksize=16))
    pickle.dump(R, open('/home/nafisnaufal1426/aciids/astar/oc4/cells.pkl', 'wb'), protocol=4)
    print('done', len(R))
