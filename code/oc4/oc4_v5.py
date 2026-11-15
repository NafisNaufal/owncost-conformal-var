"""v5 EXPLORATORY additions requested by the round-3 review (all on the 2015-2025 window already examined; unadjusted).
  paired     - buffer minus each same-feature baseline, and real minus lag-1 inputs (pooled tick loss, 21-day block bootstrap)
  uplift     - first-day null C: the tracker's VaR raised on days a SHIFTED flag is on by the real buffer's own flag-day VaR
               uplift (per target x base x level), onsets grouped by that shifted flag; and fixed uplift on the REAL flag
  enrichment - share of onsets with the flag on versus the share of all test days with the flag on
  stale      - per-target corr(open-to-close return, prior US return) with block-bootstrap intervals; tick gain and US-led
               first-day change on the targets whose interval includes zero
  canada     - Canada's tick gain with the US flag only and with all three inputs
  zones      - binomial yellow-zone benchmarks for an independent VaR at the realised violation rates
  jumps      - mean VaR ratio and day-on-day VaR changes, buffer versus tracker, both timings
Writes v5.json."""
import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/oc4')
import json
import numpy as np, pandas as pd
from scipy.stats import binom
import oc4_lib as L
import oc4_study as S
import oc4_eval as E
import oc4_event as EV
D = S.D; R = EV.R; OUT = {}
lo, hi = L.TEST
GRID = {}
for tm in ['oc', 'cc']:
    U = np.unique(np.concatenate([D[k][tm]['dates'] for k in S.TG])); GRID[tm] = (U, (U >= np.datetime64(lo)) & (U < np.datetime64(hi)))


def ongrid(tm, k, x):
    U, m = GRID[tm]; out = np.full(len(U), np.nan); out[np.searchsorted(U, D[k][tm]['dates'])] = x; return out[m]


def stack(key_fn, field, tm='oc', tg=None):
    return np.column_stack([ongrid(tm, k, R[key_fn(k, b)][field]) for k in (tg or S.TG) for b in S.BASES])


RNG = np.random.default_rng(11); NB = 3000; BLK = 21
def boot_idx(T):
    st = RNG.integers(0, T, (NB, int(np.ceil(T / BLK)))); return ((st[:, :, None] + np.arange(BLK)).reshape(NB, -1)[:, :T]) % T
IDX = {tm: boot_idx(int(GRID[tm][1].sum())) for tm in GRID}


def mean_ci(M, tm):
    s = np.nansum(M, 1); c = np.sum(np.isfinite(M), 1).astype(float); I = IDX[tm]; b = s[I].sum(1) / c[I].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def mk(tm, a, m): return lambda k, b: (tm, k, b, a, m)


# ---------- paired tick-loss differences ----------
rows = []
for a in S.LV:
    Tt = stack(mk('oc', a, 'Tracker'), 'tk'); mt = np.nanmean(Tt); Tb = stack(mk('oc', a, 'LinBuf'), 'tk')
    comps = {m: stack(mk('oc', a, m), 'tk') for m in ['PID', 'QR', 'DtACI', 'Tracker']}
    comps['lag-1 inputs (all three)'] = stack(lambda k, b: ('pl', 'X', E.MASKS['all three'], 1, k, b, a), 'tk')
    comps['lag-1 inputs (US + others)'] = stack(lambda k, b: ('pl', 'X', E.MASKS['US + others'], 1, k, b, a), 'tk')
    for name, Tm in comps.items():
        est, l, h = mean_ci(Tb - Tm, 'oc')
        rows.append(dict(alpha=a, comparison=f'buffer minus {name}', pct_of_tracker=100 * est / mt, lo=100 * l / mt, hi=100 * h / mt))
    # real-minus-lag-1 for the cross-market-only buffer (US + others), against its own real-input version
    Treal = stack(lambda k, b: ('abl', k, b, a, 'US + others'), 'tk'); Tl1 = comps['lag-1 inputs (US + others)']
    est, l, h = mean_ci(Treal - Tl1, 'oc'); rows.append(dict(alpha=a, comparison='US + others: real minus lag-1', pct_of_tracker=100 * est / mt, lo=100 * l / mt, hi=100 * h / mt))
OUT['paired'] = rows; print(pd.DataFrame(rows).round(2).to_string(index=False), flush=True)

# ---------- first-day nulls with the buffer's own uplift ----------
def flagcol(tm): return 1 if tm == 'oc' else 0


def test_mask(k, tm):
    d = D[k][tm]['dates']; return (d >= np.datetime64(lo)) & (d < np.datetime64(hi))


def uplift(k, b, a, tm):
    f = D[k][tm]['F'][:, flagcol(tm)]; vb = R[(tm, k, b, a, 'LinBuf')]['vv']; vt = R[(tm, k, b, a, 'Tracker')]['vv']
    m = (f == 1) & test_mask(k, tm) & np.isfinite(vb) & np.isfinite(vt); return float(np.mean(vb[m]) / np.mean(vt[m]))


rows = []; enr = []
for tm in ['oc', 'cc']:
    ON = {k: EV.onsets(k, tm) for k in S.TG}; fc = flagcol(tm)
    UP = {(k, b, a): uplift(k, b, a, tm) for k in S.TG for b in S.BASES for a in S.LV}
    nflag = sum(int(D[k][tm]['F'][i, fc] > 0) for k in S.TG for i in ON[k]); non = sum(len(ON[k]) for k in S.TG)
    share_days = float(np.mean(np.concatenate([D[k][tm]['F'][test_mask(k, tm), fc] for k in S.TG])))
    enr.append(dict(timing=tm, onsets=non, flagged_onsets=nflag, share_onsets_flagged=100 * nflag / non, share_days_flagged=100 * share_days, enrichment=(nflag / non) / share_days))
    for a in S.LV:
        def recs(flag_fn):
            z = []
            for k in S.TG:
                loss = D[k][tm]['loss']
                for b in S.BASES:
                    vt = R[(tm, k, b, a, 'Tracker')]['vv']; et = R[(tm, k, b, a, 'Tracker')]['e']; f = flag_fn(k)
                    for i in ON[k]:
                        if f[i] > 0:
                            vn = vt[i] * UP[(k, b, a)]; z.append(dict(k=k, b=b, date=D[k][tm]['dates'][i], tr=float(et[i]), nul=float(loss[i] > vn)))
            return pd.DataFrame(z)
        real = recs(lambda k: D[k][tm]['F'][:, fc]); d_real = 100 * (real.nul - real.tr).mean()
        vals, trs, nls = [], [], []
        for l in E.LONG:
            z = recs(lambda k, l=l: np.roll(D[k][tm]['F'][:, fc], l))
            if len(z): vals.append(100 * (z.nul - z.tr).mean()); trs.append(100 * z.tr.mean()); nls.append(100 * z.nul.mean())
        ev = [r for r in json.load(open('/home/nafisnaufal1426/aciids/astar/oc4/event.json'))[tm] if r['label'] == 'real' and r['group'] == 'US led' and r['alpha'] == a][0]
        rows.append(dict(timing=tm, alpha=a, uplift_mean=float(np.mean([UP[(k, b, a)] for k in S.TG for b in S.BASES])),
                         real_buffer_change=ev['diff'], real_tracker=ev['tracker'], real_buffer=ev['buffer'],
                         fixed_uplift_real_flag_change=d_real, fixed_uplift_real_flag_rate=100 * real.nul.mean(),
                         nullC_mean_change=float(np.mean(vals)), nullC_min=float(np.min(vals)), nullC_max=float(np.max(vals)),
                         nullC_tracker_rate=float(np.mean(trs)), nullC_rate=float(np.mean(nls)), n_le_real=int(np.sum(np.array(vals) <= ev['diff'])), n=len(vals),
                         excess_over_nullC=ev['diff'] - float(np.mean(vals))))
OUT['uplift_null'] = rows; OUT['enrichment'] = enr
print(pd.DataFrame(rows).round(2).to_string(index=False)); print(pd.DataFrame(enr).round(3).to_string(index=False), flush=True)

# ---------- stale opens: per-target correlation with intervals ----------
rows = []; rng = np.random.default_rng(4)
for k in S.TG:
    r = D[k]['oc']; dd = pd.DatetimeIndex(r['dates']); m = (dd >= lo) & (dd < hi)
    zus = r['Z']['USA'][0]; okz = (r['Z']['USA'][2] <= 3) & np.isfinite(zus) & m
    x = -r['loss'][okz]; y = zus[okz]; T = len(x); est = float(np.corrcoef(x, y)[0, 1]); bs = []
    for _ in range(2000):
        st = rng.integers(0, T, int(np.ceil(T / BLK))); ii = ((st[:, None] + np.arange(BLK)).ravel()[:T]) % T; bs.append(np.corrcoef(x[ii], y[ii])[0, 1])
    rows.append(dict(target=k, corr=est, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5))))
pc = pd.DataFrame(rows); clean = pc[(pc.lo <= 0) & (pc.hi >= 0)].target.tolist(); print(pc.round(3).to_string(index=False)); print('near-zero targets', clean, flush=True)
st_rows = []
for a in S.LV:
    for name, tg in [('all targets', S.TG), ('targets with corr interval including 0', clean)]:
        Tt = stack(mk('oc', a, 'Tracker'), 'tk', 'oc', tg); Tb = stack(mk('oc', a, 'LinBuf'), 'tk', 'oc', tg); est, l, h = mean_ci(Tb - Tt, 'oc'); mt = np.nanmean(Tt)
        st_rows.append(dict(alpha=a, set=name, n_targets=len(tg), tick_pct=100 * est / mt, lo=100 * l / mt, hi=100 * h / mt))
    ON = {k: EV.onsets(k, 'oc') for k in clean}
    z = []
    for k in clean:
        for i in ON[k]:
            if D[k]['oc']['F'][i, 1] > 0:
                for b in S.BASES:
                    z.append(dict(k=k, b=b, date=D[k]['oc']['dates'][i], buf=float(R[('oc', k, b, a, 'LinBuf')]['e'][i]), tr=float(R[('oc', k, b, a, 'Tracker')]['e'][i])))
    z = pd.DataFrame(z)
    if len(z):
        z['ep'] = EV.episodes(z); dlt = (z.buf - z.tr).to_numpy(); cl = z.ep.to_numpy()
        w = EV.wild_webb(dlt, cl) if len(np.unique(cl)) > 1 else (np.nan, np.nan, np.nan, 1)
        st_rows.append(dict(alpha=a, set='US-led first day, near-zero targets', onsets=len(z) // 2, episodes=int(z.ep.nunique()), tracker=100 * z.tr.mean(), buffer=100 * z.buf.mean(), diff=100 * dlt.mean(), lo=100 * w[0], hi=100 * w[1]))
OUT['stale_corr'] = rows; OUT['stale_subset'] = st_rows; OUT['near_zero_targets'] = clean
print(pd.DataFrame(st_rows).round(2).to_string(index=False), flush=True)

# ---------- Canada ----------
rows = []
for a in S.LV:
    Tt = stack(mk('oc', a, 'Tracker'), 'tk', 'oc', ['CAN']); mt = np.nanmean(Tt)
    for inp in ['US only', 'all three', 'own gap only']:
        Tb = stack(lambda k, b: ('abl', k, b, a, inp), 'tk', 'oc', ['CAN']); est, l, h = mean_ci(Tb - Tt, 'oc')
        rows.append(dict(alpha=a, inputs=inp, pct=100 * est / mt, lo=100 * l / mt, hi=100 * h / mt))
OUT['canada'] = rows; print(pd.DataFrame(rows).round(2).to_string(index=False), flush=True)

# ---------- yellow-zone benchmarks ----------
rows = []
for tm in ['oc', 'cc']:
    for m in ['Tracker', 'LinBuf']:
        v = float(np.nanmean(stack(mk(tm, 0.01, m), 'e', tm)))
        rows.append(dict(timing=tm, method=m, viol=100 * v, binomial_yellow_or_worse=100 * float(binom.sf(4, 250, v)), binomial_yellow=100 * float(binom.sf(4, 250, v) - binom.sf(9, 250, v))))
rows.append(dict(timing='nominal', method='independent VaR at 1%', viol=1.0, binomial_yellow_or_worse=100 * float(binom.sf(4, 250, 0.01)), binomial_yellow=100 * float(binom.sf(4, 250, 0.01) - binom.sf(9, 250, 0.01))))
OUT['zones'] = rows; print(pd.DataFrame(rows).round(2).to_string(index=False), flush=True)

# ---------- VaR level and jumps ----------
rows = []
for tm in ['oc', 'cc']:
    for a in S.LV:
        Vt = stack(mk(tm, a, 'Tracker'), 'vv', tm); Vb = stack(mk(tm, a, 'LinBuf'), 'vv', tm)
        def jumps(V):
            J = np.diff(V, axis=0) / np.abs(V[:-1]); J = J[np.isfinite(J)]; return float(np.mean(np.abs(J))), float(np.percentile(J, 99)), float(np.mean(J > 0.5))
        jt = jumps(Vt); jb = jumps(Vb)
        rows.append(dict(timing=tm, alpha=a, mean_var_ratio=float(np.nanmean(Vb) / np.nanmean(Vt)), tracker_mean_abs_change=100 * jt[0], buffer_mean_abs_change=100 * jb[0],
                         tracker_p99_rise=100 * jt[1], buffer_p99_rise=100 * jb[1], tracker_share_rise_gt50=100 * jt[2], buffer_share_rise_gt50=100 * jb[2]))
OUT['jumps'] = rows; print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)

# ---------- all-onset rows from event.json ----------
ej = json.load(open('/home/nafisnaufal1426/aciids/astar/oc4/event.json'))
OUT['all_onsets'] = [r for tm in ['oc', 'cc'] for r in ej[tm] if r['label'] == 'real' and r['group'] == 'all']
json.dump(OUT, open('/home/nafisnaufal1426/aciids/astar/oc4/v5.json', 'w'), indent=1, default=float)
print('wrote v5.json')
