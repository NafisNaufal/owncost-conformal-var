"""v4 EXPLORATORY first-day event study, both timings.
Onset: own loss above its rolling 250-day 95% quantile (prior data) after Q quiet own days, with at least a share c of the other
26 markets exceeding (close-to-close) within +-w calendar days; the co-exceedance condition uses later data and only selects
events. Groups use the US flag the buffer sees (known before the VaR is set).
Nulls (40 common lags, buffer steps re-tuned on shifted inputs):
  A 'shift-inputs': buffer fed shifted cross-market inputs, onsets grouped by the REAL flag (v3 placebo);
  B 'self-grouped': buffer fed shifted inputs AND onsets grouped by the SHIFTED flag it sees (tests conditioning on an input).
Episodes: onsets across targets chained when within 5 calendar days. Few-cluster inference: wild cluster bootstrap with Webb
6-point weights (percentile-t, 9999 draws) and a t interval with G-1 df on episode clusters; leave-one-episode-out range."""
import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/oc4')
import json, pickle, itertools
import numpy as np, pandas as pd
from scipy.stats import t as tdist
import oc4_lib as L
import oc4_study as S
import oc4_eval as E
D = S.D; R = pickle.load(open('/home/nafisnaufal1426/aciids/astar/oc4/cells.pkl', 'rb'))
data = {m: L.load_ohlc(m) for m in L.PANEL}
cc_ex = {}
for m in L.PANEL:
    l = -data[m].r; thr = l.shift(1).rolling(250).quantile(0.95); cc_ex[m] = ((l > thr) & thr.notna()).astype(float)
EXS = {m: cc_ex[m][cc_ex[m] > 0].index.values for m in L.PANEL}


def onsets(k, tm, Q=20, c=0.25, w=1):
    d = D[k][tm]['dates']; loss = pd.Series(D[k][tm]['loss']); thr = loss.shift(1).rolling(250).quantile(0.95).to_numpy(); x = (loss.to_numpy() > thr) & np.isfinite(thr)
    out = []; lo, hi = np.datetime64(L.TEST[0]), np.datetime64(L.TEST[1])
    for i in range(Q, len(d)):
        if not x[i] or x[i - Q:i].any() or not (lo <= d[i] < hi): continue
        n = 0; tot = 0; t0 = d[i] - np.timedelta64(w, 'D'); t1 = d[i] + np.timedelta64(w, 'D')
        for j in L.PANEL:
            if j == k: continue
            ix = data[j].index.values; has = np.any((ix >= t0) & (ix <= t1))
            if not has: continue
            tot += 1; ex = EXS[j]; n += int(np.any((ex >= t0) & (ex <= t1)))
        if tot and n >= c * tot: out.append(i)
    return out


def episodes(recs):
    """chain onset dates (any target) within 5 calendar days into episodes"""
    ds = np.array(sorted(set(recs.date.values))); ep = {}; cur = 0
    for i, dd in enumerate(ds):
        if i and (dd - ds[i - 1]) > np.timedelta64(5, 'D'): cur += 1
        ep[dd] = cur
    return recs.date.map(ep).to_numpy()


def wild_webb(d, cl, B=9999, seed=3):
    """wild cluster bootstrap-t for the mean of d with clusters cl (Webb 6-point weights); returns 95% percentile-t interval."""
    rng = np.random.default_rng(seed); u = np.unique(cl); G = len(u); n = len(d); m = d.mean()
    S_ = np.array([d[cl == g].sum() for g in u]); N_ = np.array([(cl == g).sum() for g in u]); res = S_ - m * N_
    def se(Sg, Ng, mm): r = Sg - mm * Ng; return np.sqrt(G / (G - 1) * np.sum(r ** 2)) / n
    s0 = se(S_, N_, m); W = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
    ts = []
    for _ in range(B):
        v = rng.choice(W, G); Sb = m * N_ + res * v; mb = Sb.sum() / n; ts.append((mb - m) / se(Sb, N_, mb))
    q = np.percentile(ts, [2.5, 97.5]); return m - q[1] * s0, m - q[0] * s0, s0, G


def t_cluster(d, cl):
    u = np.unique(cl); G = len(u); n = len(d); m = d.mean(); r = np.array([d[cl == g].sum() - m * (cl == g).sum() for g in u])
    se = np.sqrt(G / (G - 1) * np.sum(r ** 2)) / n; q = tdist.ppf(0.975, G - 1); return m - q * se, m + q * se


def first_day(tm, ON, a, buf_key, grp_flag):
    recs = []
    for k in S.TG:
        for i in ON[k]:
            for b in S.BASES:
                recs.append(dict(k=k, b=b, date=D[k][tm]['dates'][i], g=('US led' if grp_flag(k, b)[i] > 0 else 'Not US led'),
                                 buf=float(R[buf_key(k, b)]['e'][i]), tr=float(R[(tm, k, b, a, 'Tracker')]['e'][i])))
    return pd.DataFrame(recs)


def summarise(z, tm, a, label):
    out = []
    z = z.copy(); z['ep'] = episodes(z)
    for g in ['US led', 'Not US led', 'all']:
        y = z if g == 'all' else z[z.g == g]
        if not len(y): continue
        d = (y.buf - y.tr).to_numpy(); cl = y.ep.to_numpy()
        lo_w, hi_w, se, G = wild_webb(d, cl) if len(np.unique(cl)) > 1 else (np.nan, np.nan, np.nan, 1)
        lo_t, hi_t = t_cluster(d, cl) if len(np.unique(cl)) > 1 else (np.nan, np.nan)
        loo = [100 * d[cl != e].mean() for e in np.unique(cl)] if len(np.unique(cl)) > 1 else [np.nan]
        top = y.groupby('ep').size().sort_values(ascending=False)
        out.append(dict(timing=tm, alpha=a, label=label, group=g, onsets=len(y) // 2, records=len(y), episodes=int(y.ep.nunique()), top4_episode_onsets=int(top.iloc[:4].sum() // 2),
                        tracker=100 * y.tr.mean(), buffer=100 * y.buf.mean(), diff=100 * d.mean(), wild_lo=100 * lo_w, wild_hi=100 * hi_w, t_lo=100 * lo_t, t_hi=100 * hi_t, loo_min=float(np.min(loo)), loo_max=float(np.max(loo))))
    return out


if __name__ == '__main__':
    OUT = {}
    for tm in ['oc', 'cc']:
        ON = {k: onsets(k, tm) for k in S.TG}; print(tm, 'onsets', sum(len(v) for v in ON.values()), flush=True)
        fcol = 1 if tm == 'oc' else 0
        real_flag = lambda k, b, tm=tm, fcol=fcol: D[k][tm]['F'][:, fcol]
        rows = []
        for a in S.LV:
            z = first_day(tm, ON, a, lambda k, b: (tm, k, b, a, 'LinBuf'), real_flag); rows += summarise(z, tm, a, 'real')
            fam = 'X' if tm == 'oc' else 'CC'; mask = E.MASKS['all three'] if tm == 'oc' else None
            for lab in ['A shift-inputs', 'B self-grouped']:
                diffs = {g: [] for g in ['US led', 'Not US led', 'all']}
                for l in E.LONG:
                    sh_flag = (lambda k, b, l=l: np.roll(D[k][tm]['F'][:, fcol], l)) if lab.startswith('B') else real_flag
                    zz = first_day(tm, ON, a, lambda k, b, l=l: ('pl', fam, mask, l, k, b, a), sh_flag)
                    for g in diffs:
                        y = zz if g == 'all' else zz[zz.g == g]
                        if len(y): diffs[g].append(100 * (y.buf - y.tr).mean())
                for g, v in diffs.items():
                    real = [r for r in rows if r['alpha'] == a and r['label'] == 'real' and r['group'] == g][0]['diff']
                    rows.append(dict(timing=tm, alpha=a, label=lab, group=g, null_mean=float(np.mean(v)), null_min=float(np.min(v)), n_le_real=int(np.sum(np.array(v) <= real)), n=len(v), excess=real - float(np.mean(v))))
        OUT[tm] = rows; print(pd.DataFrame(rows).round(2).to_string(index=False), flush=True)
    # sensitivity grid of the onset definition (oc timing, real inputs)
    rows = []
    for Q, c, w in itertools.product([10, 20, 40], [0.1, 0.25, 0.5], [0, 1, 2]):
        ON = {k: onsets(k, 'oc', Q, c, w) for k in S.TG}
        for a in S.LV:
            z = first_day('oc', ON, a, lambda k, b: ('oc', k, b, a, 'LinBuf'), lambda k, b: D[k]['oc']['F'][:, 1])
            for g in ['US led', 'Not US led']:
                y = z[z.g == g]; rows.append(dict(Q=Q, c=c, w=w, alpha=a, group=g, onsets=len(y) // 2, diff=100 * (y.buf - y.tr).mean() if len(y) else np.nan))
    sg = pd.DataFrame(rows); OUT['sensitivity'] = rows
    print(sg.groupby(['alpha', 'group']).agg(min_diff=('diff', 'min'), max_diff=('diff', 'max'), min_n=('onsets', 'min'), max_n=('onsets', 'max')).round(2))
    json.dump(OUT, open('/home/nafisnaufal1426/aciids/astar/oc4/event.json', 'w'), indent=1, default=float)
