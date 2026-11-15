"""v6 EXPLORATORY additions requested by the round-4 review (2015-2025 window already examined; unadjusted).
  concentration - share of the buffer's net tick-loss gain over the tracker from the top 10/20/50 days, the gain without each
                  calendar year, and 2020's share (S9)
  onset_groups  - flagged vs unflagged onsets: standardised loss on the onset day, tracker first-day violation rate, and their
                  differences with episode-bootstrap intervals; onsets re-detected with the US excluded from the co-exceedance
                  count (R8, S10)
  zones_ci      - yellow-zone share difference (buffer minus tracker) with a 250-day moving-block bootstrap over window starts (S25)
  prop2_bound   - Proposition 2's bound (2B + KW + eta_u)/eta_u at the selected steps, B = max |score| per cell, in violations,
                  against alpha x test days (R11)
Writes v6.json."""
import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/oc4')
import json
import numpy as np, pandas as pd
import oc4_lib as L
import oc4_study as S
import oc4_event as EV
D = S.D; R = EV.R; OUT = {}
lo, hi = L.TEST
U = np.unique(np.concatenate([D[k]['oc']['dates'] for k in S.TG])); MK = (U >= np.datetime64(lo)) & (U < np.datetime64(hi)); UD = U[MK]


def og(k, x):
    o = np.full(len(U), np.nan); o[np.searchsorted(U, D[k]['oc']['dates'])] = x; return o[MK]


def stack(m, a, field):
    return np.column_stack([og(k, R[('oc', k, b, a, m)][field]) for k in S.TG for b in S.BASES])


# ---------- concentration ----------
rows = []
for a in S.LV:
    Tt = stack('Tracker', a, 'tk'); Tb = stack('LinBuf', a, 'tk'); dlt = Tb - Tt
    num_t = np.nansum(dlt, 1); cnt = np.sum(np.isfinite(dlt), 1); tot = num_t.sum(); mt = np.nanmean(Tt)
    pct = 100 * (tot / cnt.sum()) / mt
    order = np.argsort(num_t)  # most negative (largest gain) first
    share = {f'top{k}': float(num_t[order[:k]].sum() / tot) for k in (10, 20, 50)}
    yrs = pd.DatetimeIndex(UD).year.values; loyo = {}
    for y in np.unique(yrs):
        mm = yrs != y; loyo[int(y)] = float(100 * (num_t[mm].sum() / cnt[mm].sum()) / np.nanmean(Tt[mm]))
    rows.append(dict(alpha=a, pct=float(pct), **share, share_2020=float(num_t[yrs == 2020].sum() / tot), without_year=loyo,
                     top_days=[str(UD[i])[:10] for i in order[:10]]))
OUT['concentration'] = rows
for r in rows: print('conc', r['alpha'], round(r['pct'], 2), {k: round(v, 3) for k, v in r.items() if k.startswith('top') and k != 'top_days'}, 'share2020', round(r['share_2020'], 3), 'loyo', {k: round(v, 2) for k, v in r['without_year'].items()}, flush=True)

# ---------- onset groups ----------
def groups(ON, a=0.01, label='paper rule'):
    recs = []
    for k in S.TG:
        sig = D[k]['oc']['sig']; loss = D[k]['oc']['loss']; F = D[k]['oc']['F'][:, 1]
        for i in ON[k]:
            for b in S.BASES:
                recs.append(dict(k=k, b=b, date=D[k]['oc']['dates'][i], flag=int(F[i] > 0), z=float(loss[i] / sig[i]),
                                 tr=float(R[('oc', k, b, a, 'Tracker')]['e'][i]), buf=float(R[('oc', k, b, a, 'LinBuf')]['e'][i])))
    z = pd.DataFrame(recs); z['ep'] = EV.episodes(z)
    rng = np.random.default_rng(21); eps = z.ep.unique()
    def diff(zz, col): return zz[zz.flag == 1][col].mean() - zz[zz.flag == 0][col].mean()
    bz, bt = [], []
    for _ in range(4000):
        pick = rng.choice(eps, len(eps)); zz = pd.concat([z[z.ep == e] for e in pick])
        if zz.flag.sum() == 0 or (zz.flag == 0).sum() == 0: continue
        bz.append(diff(zz, 'z')); bt.append(diff(zz, 'tr'))
    n_on = len(z) // 2; n_fl = int(z.flag.sum() // 2)
    return dict(rule=label, onsets=n_on, flagged=n_fl, flagged_share=100 * n_fl / n_on, episodes=int(len(eps)),
                z_flagged=float(z[z.flag == 1].z.mean()), z_unflagged=float(z[z.flag == 0].z.mean()), z_diff=float(diff(z, 'z')), z_lo=float(np.percentile(bz, 2.5)), z_hi=float(np.percentile(bz, 97.5)),
                tr_flagged=100 * float(z[z.flag == 1].tr.mean()), tr_unflagged=100 * float(z[z.flag == 0].tr.mean()), tr_diff=100 * float(diff(z, 'tr')),
                tr_lo=100 * float(np.percentile(bt, 2.5)), tr_hi=100 * float(np.percentile(bt, 97.5)), buf_flagged=100 * float(z[z.flag == 1].buf.mean()), buf_unflagged=100 * float(z[z.flag == 0].buf.mean()))
ON = {k: EV.onsets(k, 'oc') for k in S.TG}
PANEL0 = list(L.PANEL); L.PANEL[:] = [m for m in PANEL0 if m != 'USA']
ON_noUS = {k: EV.onsets(k, 'oc') for k in S.TG}
L.PANEL[:] = PANEL0
OUT['onset_groups'] = [groups(ON), groups(ON_noUS, label='US excluded from co-exceedance')]
print(pd.DataFrame(OUT['onset_groups']).round(2).to_string(index=False), flush=True)

# ---------- yellow-zone interval ----------
rows = []
rng = np.random.default_rng(31)
Xt = stack('Tracker', 0.01, 'e'); Xb = stack('LinBuf', 0.01, 'e')
def yel(X):
    Y = np.full((X.shape[0] - 249, X.shape[1]), np.nan)
    for j in range(X.shape[1]):
        x = X[:, j]; ok = np.isfinite(x)
        if ok.sum() < 250: continue
        c = np.convolve(np.where(ok, x, 0), np.ones(250), 'valid'); n = np.convolve(ok.astype(float), np.ones(250), 'valid')
        Y[:, j] = np.where(n >= 240, ((c >= 5) & (c <= 9)).astype(float), np.nan)
    return Y
Yt, Yb = yel(Xt), yel(Xb); dY = np.nansum(Yb - Yt, 1); nY = np.sum(np.isfinite(Yb - Yt), 1); Tn = len(dY)
est = dY.sum() / nY.sum(); bs = []
for _ in range(3000):
    st = rng.integers(0, Tn, int(np.ceil(Tn / 250))); I = ((st[:, None] + np.arange(250)).ravel()[:Tn]) % Tn; bs.append(dY[I].sum() / nY[I].sum())
OUT['zones_ci'] = dict(yellow_tracker=100 * float(np.nanmean(Yt)), yellow_buffer=100 * float(np.nanmean(Yb)), diff=100 * float(est), lo=100 * float(np.percentile(bs, 2.5)), hi=100 * float(np.percentile(bs, 97.5)))
print('zones', {k: round(v, 2) for k, v in OUT['zones_ci'].items()}, flush=True)

# ---------- Proposition 2 bound ----------
import pickle
sel = pickle.load(open('/home/nafisnaufal1426/aciids/astar/oc4/selected.pkl', 'rb'))
rows = []
for a in S.LV:
    eta_u = sel[('oc', a, 'LinBuf')]['eta_u']; Bs, bounds, ntest = [], [], []
    for k in S.TG:
        for b in S.BASES:
            s, v, ok = S.scores(k, b, a, 'oc'); i0 = int(np.argmax(ok)); B = float(np.nanmax(np.abs(s[i0:]))); Bs.append(B)
            bounds.append((2 * B + 3 * 6.0 + eta_u) / eta_u); d = D[k]['oc']['dates']; ntest.append(int(((d >= np.datetime64(lo)) & (d < np.datetime64(hi))).sum()))
    rows.append(dict(alpha=a, eta_u=eta_u, B_median=float(np.median(Bs)), B_max=float(np.max(Bs)), bound_min=float(np.min(bounds)), bound_median=float(np.median(bounds)), bound_max=float(np.max(bounds)),
                     test_days_median=float(np.median(ntest)), expected_violations=float(a * np.median(ntest))))
OUT['prop2_bound'] = rows; print(pd.DataFrame(rows).round(1).to_string(index=False), flush=True)
json.dump(OUT, open('/home/nafisnaufal1426/aciids/astar/oc4/v6.json', 'w'), indent=1, default=float)
print('wrote v6.json')
