"""v7 EXPLORATORY additions requested by the round-5 review (2015-2025 window already examined; unadjusted).
  open_quality - per-target share of 2015-2025 days on which the open equals the day's high or low (a stale-open marker),
                 and the buffer's tick-loss gain over the tracker for the targets below and above the median share (R7)
  zones_cc     - yellow-zone share at 1% with VaR at the previous close (supervisory timing), buffer minus tracker, with the
                 same moving-block bootstrap as oc4_v6b.py (S25)
  target_rule  - share of 2005-2025 days on which the open differs from the previous close, for every panel market, with the
                 95% rule used to choose the 17 targets (stated in astar/oc/oc_lib.py), applied in code (S31)
Writes v7.json."""
import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/oc4')
import json, pickle
import numpy as np, pandas as pd
import oc4_lib as L
import oc4_study as S
D = S.D; R = pickle.load(open('/home/nafisnaufal1426/aciids/astar/oc4/cells.pkl', 'rb')); OUT = {}
lo, hi = L.TEST; RNG = np.random.default_rng(71)


def grid(tm):
    U = np.unique(np.concatenate([D[k][tm]['dates'] for k in S.TG])); M = (U >= np.datetime64(lo)) & (U < np.datetime64(hi)); return U, M


def og(tm, k, x):
    U, M = GR[tm]; o = np.full(len(U), np.nan); o[np.searchsorted(U, D[k][tm]['dates'])] = x; return o[M]


GR = {tm: grid(tm) for tm in ['oc', 'cc']}


def pct_ci(tg, a, tm='oc', blk=21, nb=3000):
    Tt = np.column_stack([og(tm, k, R[(tm, k, b, a, 'Tracker')]['tk']) for k in tg for b in S.BASES])
    Tb = np.column_stack([og(tm, k, R[(tm, k, b, a, 'LinBuf')]['tk']) for k in tg for b in S.BASES])
    d = Tb - Tt; num = np.nansum(d, 1); cnt = np.sum(np.isfinite(d), 1); mt = np.nanmean(Tt); T = len(num)
    st = RNG.integers(0, T, (nb, int(np.ceil(T / blk)))); I = ((st[:, :, None] + np.arange(blk)).reshape(nb, -1)[:, :T]) % T
    bs = num[I].sum(1) / cnt[I].sum(1)
    return 100 * num.sum() / cnt.sum() / mt, 100 * np.percentile(bs, 2.5) / mt, 100 * np.percentile(bs, 97.5) / mt


# ---------- open quality ----------
share = {}
for k in S.TG:
    d = L.load_ohlc(k); d = d.loc[lo:hi]; ok = d.open.notna() & d.high.notna() & d.low.notna()
    share[k] = float(((d.open == d.high) | (d.open == d.low))[ok].mean()) if ok.any() else float('nan')
med = float(np.nanmedian(list(share.values())))
low = [k for k in S.TG if share[k] <= med]; high = [k for k in S.TG if share[k] > med]
rows = []
for a in S.LV:
    for name, tg in [('low open-is-extreme share', low), ('high open-is-extreme share', high), ('all', S.TG)]:
        e, l, h = pct_ci(tg, a); rows.append(dict(alpha=a, group=name, n_targets=len(tg), pct=float(e), lo=float(l), hi=float(h)))
OUT['open_quality'] = dict(share_2015_2025=share, median=med, low=low, high=high, gains=rows)
print('shares', {k: round(v, 3) for k, v in share.items()}, 'median', round(med, 3)); print(pd.DataFrame(rows).round(2).to_string(index=False), flush=True)

# ---------- yellow zone at the previous close ----------
U, M = GR['cc']; UD = U[M]; num = np.zeros(len(UD)); cnt = np.zeros(len(UD)); yt = yb = nw = 0
for k in S.TG:
    for b in S.BASES:
        xt = og('cc', k, R[('cc', k, b, 0.01, 'Tracker')]['e']); xb = og('cc', k, R[('cc', k, b, 0.01, 'LinBuf')]['e']); ok = np.isfinite(xt) & np.isfinite(xb)
        it = np.where(ok)[0]; wt = np.convolve(xt[ok].astype(int), np.ones(250, int), 'valid'); wb = np.convolve(xb[ok].astype(int), np.ones(250, int), 'valid')
        Yt = (wt >= 5) & (wt <= 9); Yb = (wb >= 5) & (wb <= 9); end = it[249:]
        np.add.at(num, end, Yb.astype(float) - Yt.astype(float)); np.add.at(cnt, end, 1.0); yt += Yt.sum(); yb += Yb.sum(); nw += len(Yt)
Tn = len(num); bs = []
for _ in range(3000):
    st = RNG.integers(0, Tn, int(np.ceil(Tn / 250))); I = ((st[:, None] + np.arange(250)).ravel()[:Tn]) % Tn; bs.append(num[I].sum() / cnt[I].sum())
OUT['zones_cc'] = dict(yellow_tracker=100 * yt / nw, yellow_buffer=100 * yb / nw, diff=100 * num.sum() / cnt.sum(), lo=100 * float(np.percentile(bs, 2.5)), hi=100 * float(np.percentile(bs, 97.5)), blocks=int(np.ceil(Tn / 250)))
print('zones_cc', {k: round(v, 2) for k, v in OUT['zones_cc'].items()}, flush=True)

# ---------- target rule ----------
rule = {}
for m in L.PANEL:
    d = L.load_ohlc(m).loc['2005-01-01':'2025-12-31']; prev = d.close.shift(1); ok = d.open.notna() & prev.notna()
    rule[m] = float((d.open[ok] != prev[ok]).mean())
sel = sorted([m for m, v in rule.items() if v > 0.95 and m != 'USA'])
OUT['target_rule'] = dict(window='2005-01-01..2025-12-31', share_open_differs=rule, selected=sel, matches_code=sorted(S.TG) == sel)
print('rule', {k: round(v, 3) for k, v in sorted(rule.items(), key=lambda x: -x[1])}, 'selected==TG', sorted(S.TG) == sel, flush=True)
json.dump(OUT, open('/home/nafisnaufal1426/aciids/astar/oc4/v7.json', 'w'), indent=1, default=float)
print('wrote v7.json')
