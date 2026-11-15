"""v6 EXPLORATORY follow-ups (round-4 R4, S25), written after oc4_v6.py so that its hashed outputs stay unchanged.
  zones_ci   - yellow-zone share at 1% exactly as oc4_analysis.py builds it (per cell, valid days only, 250-day windows), buffer
               minus tracker, with a moving-block bootstrap (block 250) over window end dates pooled across cells
  flag_cont  - open-to-close return standardised by EWMA volatility on US-flag days versus other days, on the nine targets
               whose open-to-close return is unrelated to the prior US return (v5 'near_zero_targets'), episode-free day
               bootstrap (21-day blocks); a stale open would show continuation (negative standardised loss sign flipped)
Writes v6b.json."""
import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/oc4')
import json, pickle
import numpy as np
import oc4_lib as L
import oc4_study as S
D = S.D; R = pickle.load(open('/home/nafisnaufal1426/aciids/astar/oc4/cells.pkl', 'rb')); OUT = {}
lo, hi = L.TEST; U = np.unique(np.concatenate([D[k]['oc']['dates'] for k in S.TG])); MK = (U >= np.datetime64(lo)) & (U < np.datetime64(hi)); UD = U[MK]
def og(k, x): o = np.full(len(U), np.nan); o[np.searchsorted(U, D[k]['oc']['dates'])] = x; return o[MK]
# zones
num = np.zeros(len(UD)); cnt = np.zeros(len(UD)); yt = yb = nw = 0
for k in S.TG:
    for b in S.BASES:
        xt = og(k, R[('oc', k, b, 0.01, 'Tracker')]['e']); xb = og(k, R[('oc', k, b, 0.01, 'LinBuf')]['e']); ok = np.isfinite(xt) & np.isfinite(xb)
        it = np.where(ok)[0]; wt = np.convolve(xt[ok].astype(int), np.ones(250, int), 'valid'); wb = np.convolve(xb[ok].astype(int), np.ones(250, int), 'valid')
        Yt = (wt >= 5) & (wt <= 9); Yb = (wb >= 5) & (wb <= 9); end = it[249:]
        np.add.at(num, end, Yb.astype(float) - Yt.astype(float)); np.add.at(cnt, end, 1.0); yt += Yt.sum(); yb += Yb.sum(); nw += len(Yt)
rng = np.random.default_rng(41); Tn = len(num); bs = []
for _ in range(3000):
    st = rng.integers(0, Tn, int(np.ceil(Tn / 250))); I = ((st[:, None] + np.arange(250)).ravel()[:Tn]) % Tn; bs.append(num[I].sum() / cnt[I].sum())
OUT['zones_ci'] = dict(yellow_tracker=100 * yt / nw, yellow_buffer=100 * yb / nw, diff=100 * num.sum() / cnt.sum(), lo=100 * float(np.percentile(bs, 2.5)), hi=100 * float(np.percentile(bs, 97.5)))
print('zones', {k: round(v, 2) for k, v in OUT['zones_ci'].items()})
# flag-day continuation on the nine near-zero targets
v5 = json.load(open('/home/nafisnaufal1426/aciids/astar/oc4/v5.json')); NZ = v5['near_zero_targets']
Zf, Zo = [], []
for k in NZ:
    r = D[k]['oc']; z = og(k, -r['loss'] / r['sig']); f = og(k, r['F'][:, 1]); Zf.append(np.where(f > 0, z, np.nan)); Zo.append(np.where(f == 0, z, np.nan))
Zf = np.column_stack(Zf); Zo = np.column_stack(Zo); usr = None
sf, nf = np.nansum(Zf, 1), np.sum(np.isfinite(Zf), 1); so, no = np.nansum(Zo, 1), np.sum(np.isfinite(Zo), 1)
est_f, est_o = sf.sum() / nf.sum(), so.sum() / no.sum(); bs = []
for _ in range(3000):
    st = rng.integers(0, Tn, int(np.ceil(Tn / 21))); I = ((st[:, None] + np.arange(21)).ravel()[:Tn]) % Tn
    bs.append(sf[I].sum() / nf[I].sum() - so[I].sum() / no[I].sum())
OUT['flag_cont'] = dict(targets=NZ, oc_z_flag_days=float(est_f), oc_z_other_days=float(est_o), diff=float(est_f - est_o), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), flag_days=int(nf.sum()))
print('flag_cont', OUT['flag_cont'])
json.dump(OUT, open('/home/nafisnaufal1426/aciids/astar/oc4/v6b.json', 'w'), indent=1, default=float)
