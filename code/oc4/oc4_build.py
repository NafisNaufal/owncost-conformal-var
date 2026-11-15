"""v4 build. Two timings per target:
  oc: open-to-close loss, VaR at the official local open; sources whose late close precedes the open.
  cc: close-to-close loss, VaR at the official close of the previous local session; sources whose late close precedes it.
Base VaRs (EWMA-normal, HS-250) use prior data only. Features use one stale rule (age <= 3 days to the reference date)."""
import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/oc4')
import pickle
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.stats import norm
import oc4_lib as L
data = {m: L.load_ohlc(m) for m in L.PANEL}
TG = L.FRESH_T + L.SEA_T; BURN = 250


def features(Z, own=None):
    ok = {j: Z[j][2] <= L.MAXAGE for j in Z}
    xus = np.where(ok['USA'], Z['USA'][1], 0.0)
    share = np.nanmean(np.array([np.where(ok[j], Z[j][1], np.nan) for j in Z]), 0); share = np.nan_to_num(share)
    zus2 = np.where(ok['USA'], Z['USA'][0] ** 2, np.nan)
    glob = np.nanmean(np.array([np.where(ok[j], Z[j][0] ** 2, np.nan) for j in Z]), 0)
    cols = ([np.minimum(np.nan_to_num(own) / 9.0, 1.0)] if own is not None else []) + [xus, share]
    return np.column_stack(cols), np.nan_to_num(zus2), np.nan_to_num(glob)


def build(k):
    d0 = data[k]; out = {}
    # open-to-close
    d = d0[d0.oc.notna() & d0.gap.notna()].copy(); dates = d.index.values
    oc = d.oc.to_numpy(); loss = -oc; sig = L.ewma_sigma(oc); ref = L.utc_at(k, dates, 'open')
    Z = L.side_info(k, data, ref, dates); gapz = (d.gap / d.sig_cc).to_numpy(); F, us2, glob = features(Z, gapz ** 2)
    rec = dict(dates=dates, loss=loss, sig=sig, gapz=gapz, Z=Z, F=F, us2=us2, glob=glob, own2=np.nan_to_num(gapz ** 2),
               openq=dict(o_eq_hl=float(np.mean((d.open == d.high) | (d.open == d.low))) if d.high.notna().any() else np.nan))
    for a in (0.01, 0.05):
        rec[('vhat', 'EWMA', a)] = sig * norm.ppf(1 - a); rec[('vhat', 'HS', a)] = pd.Series(loss).shift(1).rolling(250).quantile(1 - a).to_numpy()
    out['oc'] = rec
    # previous-close timing, close-to-close loss
    d = d0[d0.r.notna()].copy(); dates = d.index.values[1:]; prev = d.index.values[:-1]
    r = d.r.to_numpy()[1:]; loss = -r; sig = d.sig_cc.to_numpy()[1:]; ref = L.utc_at(k, prev, 'close')
    Z = L.side_info(k, data, ref, prev); F, us2, glob = features(Z, None)
    rec = dict(dates=dates, loss=loss, sig=sig, Z=Z, F=F, us2=us2, glob=glob)
    for a in (0.01, 0.05):
        rec[('vhat', 'EWMA', a)] = sig * norm.ppf(1 - a); rec[('vhat', 'HS', a)] = pd.Series(loss).shift(1).rolling(250).quantile(1 - a).to_numpy()
    out['cc'] = rec
    return k, out


def qr_job(t):
    k, b, a = t; r = D[k]['oc']; v = r[('vhat', b, a)]; s = (r['loss'] - v) / r['sig']; ok = np.isfinite(s); i0 = int(np.argmax(ok))
    sh = np.full(len(s), np.nan); sh[i0:] = L.qr_scorecaster(s[i0:], r['F'][i0:], a, BURN)
    return (k, b, a), sh


if __name__ == '__main__':
    with Pool(len(TG)) as p:
        D = dict(p.map(build, TG))
    with Pool(68) as p:
        Q = dict(p.map(qr_job, [(k, b, a) for k in TG for b in ['EWMA', 'HS'] for a in (0.01, 0.05)]))
    for (k, b, a), sh in Q.items():
        D[k]['oc'][('shat', b, a)] = sh
    pickle.dump(D, open('/home/nafisnaufal1426/aciids/astar/oc4/oc4_data.pkl', 'wb'))
    for k in TG:
        r = D[k]['oc']; c = D[k]['cc']
        print(k, len(r['dates']), str(r['dates'][0])[:10], 'US age med', np.nanmedian(r['Z']['USA'][2]), 'US flag share %.3f' % r['F'][:, 1].mean(),
              '| cc', len(c['dates']), 'US age med', np.nanmedian(c['Z']['USA'][2]), 'flag %.3f' % c['F'][:, 0].mean(), '| open=H/L %.3f' % r['openq']['o_eq_hl'])
