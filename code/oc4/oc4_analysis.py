"""v4 EXPLORATORY analysis of cells.pkl -> analysis.json. All intervals: 21-day circular block bootstrap over dates (3000 draws),
pooling every target x base column on a common date grid; unadjusted for multiplicity."""
import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/oc4')
import json, pickle
import numpy as np, pandas as pd
from scipy.stats import chi2, spearmanr
import oc4_lib as L
import oc4_study as S
import oc4_eval as E
D = S.D; R = pickle.load(open('/home/nafisnaufal1426/aciids/astar/oc4/cells.pkl', 'rb')); OUT = {}
lo, hi = L.TEST
GRID = {}
for tm in ['oc', 'cc']:
    U = np.unique(np.concatenate([D[k][tm]['dates'] for k in S.TG])); GRID[tm] = (U, (U >= np.datetime64(lo)) & (U < np.datetime64(hi)))


def ongrid(tm, k, x):
    U, m = GRID[tm]; out = np.full(len(U), np.nan); out[np.searchsorted(U, D[k][tm]['dates'])] = x; return out[m]


def stack(key_fn, field, tm='oc', tg=None):
    tg = tg or S.TG
    return np.column_stack([ongrid(tm, k, R[key_fn(k, b)][field]) for k in tg for b in S.BASES])


def colstack(tm, fn, tg=None):
    tg = tg or S.TG
    return np.column_stack([ongrid(tm, k, fn(k, b)) for k in tg for b in S.BASES])


RNG = np.random.default_rng(1); NB = 3000; BLK = 21
def boot_idx(T):
    st = RNG.integers(0, T, (NB, int(np.ceil(T / BLK)))); return ((st[:, :, None] + np.arange(BLK)).reshape(NB, -1)[:, :T]) % T
IDX = {tm: boot_idx(int(GRID[tm][1].sum())) for tm in GRID}


def ratio_ci(num, den, tm):
    """mean of num over finite cells / mean of den, block-bootstrap over dates. num, den: (T, C) arrays (den may be None = count)."""
    s = np.nansum(num, 1); c = np.sum(np.isfinite(num), 1).astype(float) if den is None else np.nansum(den, 1)
    est = s.sum() / c.sum(); I = IDX[tm]; b = s[I].sum(1) / c[I].sum(1)
    return float(est), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def mkey(tm, a, m): return lambda k, b: (tm, k, b, a, m)


# ---------- main tables ----------
for tm, meths in [('oc', E.OCM), ('cc', E.CCM)]:
    rows = []
    for a in S.LV:
        Tt = stack(mkey(tm, a, 'Tracker'), 'tk', tm); Vt = stack(mkey(tm, a, 'Tracker'), 'vv', tm)
        for m in meths:
            T = stack(mkey(tm, a, m), 'tk', tm); Ev = stack(mkey(tm, a, m), 'e', tm); V = stack(mkey(tm, a, m), 'vv', tm)
            est, l, h = ratio_ci(T - Tt, None, tm)
            rows.append(dict(alpha=a, method=m, viol=100 * np.nanmean(Ev), tick=100 * np.nanmean(T), d=100 * est, lo=100 * l, hi=100 * h,
                             pct=100 * est / np.nanmean(Tt), var_ratio=float(np.nanmean(V) / np.nanmean(Vt))))
    OUT[f'main_{tm}'] = rows; print(tm); print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------- ablation ----------
rows = []
for a in S.LV:
    Tt = stack(mkey('oc', a, 'Tracker'), 'tk')
    for mk in E.MASKS:
        T = stack(lambda k, b: ('abl', k, b, a, mk), 'tk'); est, l, h = ratio_ci(T - Tt, None, 'oc')
        rows.append(dict(alpha=a, inputs=mk, pct=100 * est / np.nanmean(Tt), lo=100 * l / np.nanmean(Tt), hi=100 * h / np.nanmean(Tt)))
OUT['ablation'] = rows; print(pd.DataFrame(rows).round(2).to_string(index=False))

# ---------- placebos ----------
def ptick(kind, f, m, l, a, tm):
    return 100 * np.nanmean(stack(lambda k, b: (kind, f, m, l, k, b, a), 'tk', tm))
rows = []
for a in S.LV:
    for f, m, tm, real_key in [('US', None, 'oc', ('oc', 'LinBuf')), ('X', E.MASKS['all three'], 'oc', ('abl', 'all three')), ('X', E.MASKS['US + others'], 'oc', ('abl', 'US + others')), ('CC', None, 'cc', ('cc', 'LinBuf'))]:
        if real_key[0] == 'abl': real = 100 * np.nanmean(stack(lambda k, b: ('abl', k, b, a, real_key[1]), 'tk'))
        else: real = 100 * np.nanmean(stack(mkey(real_key[0], a, real_key[1]), 'tk', tm))
        trk = 100 * np.nanmean(stack(mkey(tm, a, 'Tracker'), 'tk', tm))
        for kind in ['pl', 'plfix']:
            vals = np.array([ptick(kind, f, m, l, a, tm) for l in E.LONG])
            rows.append(dict(alpha=a, family=f, inputs=('US flag only' if f == 'US' else 'previous-close, US + others' if f == 'CC' else ('all three' if m == (1, 1, 1) else 'US + others')),
                             steps='re-tuned' if kind == 'pl' else 'real-tuned', real=real, tracker=trk, mean=float(vals.mean()), min=float(vals.min()), n_le_real=int((vals <= real).sum()), n=len(vals)))
            if f == 'X':
                for l in E.SHORT:
                    v = ptick(kind, f, m, l, a, tm); rows[-1][f'lag{l}'] = v
    vals = np.array([100 * np.nanmean(stack(lambda k, b: ('plgct', l, k, b, a), 'tk')) for l in E.LONG])
    real = 100 * np.nanmean(stack(mkey('oc', a, 'GCT-us'), 'tk'))
    rows.append(dict(alpha=a, family='US', inputs='GCT on US stream', steps='real-tuned', real=real, tracker=100 * np.nanmean(stack(mkey('oc', a, 'Tracker'), 'tk')), mean=float(vals.mean()), min=float(vals.min()), n_le_real=int((vals <= real).sum()), n=len(vals)))
OUT['placebo'] = rows; print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------- standard backtests ----------
def kupiec(x, a):
    n = len(x); k = x.sum(); ph = k / n
    if k in (0, n): return 2 * (-(n - k) * np.log(1 - a) - k * np.log(a) + (n * np.log(1 - ph) if k == 0 else n * np.log(ph)))
    return -2 * ((n - k) * np.log(1 - a) + k * np.log(a) - (n - k) * np.log(1 - ph) - k * np.log(ph))
def chr_ind(x):
    x0, x1 = x[:-1], x[1:]; n00 = np.sum((x0 == 0) & (x1 == 0)); n01 = np.sum((x0 == 0) & (x1 == 1)); n10 = np.sum((x0 == 1) & (x1 == 0)); n11 = np.sum((x0 == 1) & (x1 == 1))
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1); p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    def ll(p, a, b): return (a * np.log(1 - p) if a else 0) + (b * np.log(p) if b else 0) if 0 < p < 1 else 0.0
    return -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
rows = []
for tm, meths in [('oc', E.OCM), ('cc', E.CCM)]:
    for a in S.LV:
        for m in meths:
            kr = ir = 0; n = 0; yel = red = nw = 0; vio_after = []; cc_ = 0
            for k in S.TG:
                for b in S.BASES:
                    x = ongrid(tm, k, R[(tm, k, b, a, m)]['e']); x = x[np.isfinite(x)].astype(int); n += 1
                    kr += kupiec(x, a) > chi2.ppf(0.95, 1); ir += chr_ind(x) > chi2.ppf(0.95, 1)
                    if a == 0.01:
                        w = np.convolve(x, np.ones(250, int), 'valid'); nw += len(w); yel += np.sum((w >= 5) & (w <= 9)); red += np.sum(w >= 10)
                    vio_after.append((x[:-1] * x[1:]).sum() / max(x[:-1].sum(), 1))
            rows.append(dict(timing=tm, alpha=a, method=m, kupiec_rej=kr, ind_rej=ir, n=n, yellow=(100 * yel / nw if nw else None), red=(100 * red / nw if nw else None), viol_after_viol=100 * float(np.mean(vio_after))))
OUT['backtests'] = rows; print(pd.DataFrame(rows).round(2).to_string(index=False))

# ---------- all days by US flag status (oc) ----------
def flag(k, b): return D[k]['oc']['F'][:, 1]
def local_ex(k, b):
    l = pd.Series(D[k]['oc']['loss']); thr = l.shift(1).rolling(250).quantile(0.95); return ((l > thr) & thr.notna()).astype(float).to_numpy()
FL = colstack('oc', flag); LX = colstack('oc', local_ex); ok = np.isfinite(FL)
rows = []
for a in S.LV:
    Et = stack(mkey('oc', a, 'Tracker'), 'e'); Eb = stack(mkey('oc', a, 'LinBuf'), 'e'); Tt = stack(mkey('oc', a, 'Tracker'), 'tk'); Tb = stack(mkey('oc', a, 'LinBuf'), 'tk')
    Vt = stack(mkey('oc', a, 'Tracker'), 'vv'); Vb = stack(mkey('oc', a, 'LinBuf'), 'vv'); net = np.nansum(Tb - Tt)
    for g, msk in [('US flag on', FL == 1), ('US flag off', FL == 0), ('flag on, no local exceedance (false alarm)', (FL == 1) & (LX == 0)), ('flag on, local exceedance', (FL == 1) & (LX == 1)), ('flag off, local exceedance', (FL == 0) & (LX == 1)), ('flag off, no local exceedance', (FL == 0) & (LX == 0))]:
        valid = msk & np.isfinite(Et) & np.isfinite(Eb)
        def sel_(X): Y = np.where(valid, X, np.nan); return Y
        share = ratio_ci(np.where(np.isfinite(Et), valid.astype(float), np.nan), None, 'oc')
        vt = ratio_ci(sel_(Et), None, 'oc'); vb = ratio_ci(sel_(Eb), None, 'oc'); dv = ratio_ci(sel_(Eb - Et), None, 'oc')
        rows.append(dict(alpha=a, group=g, share=100 * share[0], viol_tracker=100 * vt[0], viol_buffer=100 * vb[0], dviol=100 * dv[0], dviol_lo=100 * dv[1], dviol_hi=100 * dv[2],
                         var_ratio=float(np.nansum(sel_(Vb)) / np.nansum(sel_(Vt))), gain_share=float(np.nansum(sel_(Tb - Tt)) / net)))
    fa = ratio_ci(np.where(FL == 1, 1 - LX, np.nan), None, 'oc'); rows.append(dict(alpha=a, group='false-alarm rate P(no local exceedance | flag on)', share=100 * fa[0], lo=100 * fa[1], hi=100 * fa[2]))
    hr = ratio_ci(np.where(LX == 1, FL, np.nan), None, 'oc'); rows.append(dict(alpha=a, group='hit rate P(flag on | local exceedance)', share=100 * hr[0], lo=100 * hr[1], hi=100 * hr[2]))
OUT['flag_days'] = rows; print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------- calm-period dependence of the score on the flag (S21) ----------
def calm(k, b):
    x = local_ex(k, b); c = pd.Series(x).shift(1).rolling(20, min_periods=20).sum().to_numpy(); return (c == 0).astype(float)
CL = colstack('oc', calm); rows = []
for a in S.LV:
    Et = stack(mkey('oc', a, 'Tracker'), 'e')
    on = ratio_ci(np.where((CL == 1) & (FL == 1), Et, np.nan), None, 'oc'); off = ratio_ci(np.where((CL == 1) & (FL == 0), Et, np.nan), None, 'oc')
    I = IDX['oc']
    def rate(M):
        s = np.nansum(M, 1); c = np.sum(np.isfinite(M), 1); return s, c
    s1, c1 = rate(np.where((CL == 1) & (FL == 1), Et, np.nan)); s0, c0 = rate(np.where((CL == 1) & (FL == 0), Et, np.nan))
    b = s1[I].sum(1) / c1[I].sum(1) - s0[I].sum(1) / c0[I].sum(1)
    rows.append(dict(alpha=a, calm_flag_on=100 * on[0], calm_flag_off=100 * off[0], diff=100 * (on[0] - off[0]), lo=100 * np.percentile(b, 2.5), hi=100 * np.percentile(b, 97.5), calm_share_flag_on=100 * float(np.nanmean(np.where(CL == 1, FL, np.nan)))))
OUT['calm_dependence'] = rows; print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------- per target and open quality ----------
rows = []
data = {m: L.load_ohlc(m) for m in L.PANEL}
for k in S.TG:
    r = D[k]['oc']; dd = pd.DatetimeIndex(r['dates']); m = (dd >= lo) & (dd < hi)
    zus = r['Z']['USA'][0]; okz = (r['Z']['USA'][2] <= 3) & np.isfinite(zus) & m
    corr = float(np.corrcoef(-r['loss'][okz], zus[okz])[0, 1])
    d = data[k].loc[(data[k].index >= lo) & (data[k].index < hi)]; oeq = float(np.mean((d.open == d.high) | (d.open == d.low)))
    row = dict(target=k, vendor=L.VENDOR[k], open_eq_high_or_low=100 * oeq, corr_oc_prior_us=corr)
    for a in S.LV:
        Tt = np.nanmean([np.nanmean(ongrid('oc', k, R[('oc', k, b, a, 'Tracker')]['tk'])) for b in S.BASES]); Tb = np.nanmean([np.nanmean(ongrid('oc', k, R[('oc', k, b, a, 'LinBuf')]['tk'])) for b in S.BASES])
        row[f'gain_{a}'] = 100 * (Tb - Tt) / Tt
    rows.append(row)
pt = pd.DataFrame(rows); print(pt.round(3).to_string(index=False))
for a in S.LV:
    for col in ['open_eq_high_or_low', 'corr_oc_prior_us']:
        rho = spearmanr(pt[col], pt[f'gain_{a}']); print('spearman', a, col, round(rho.correlation, 3), round(rho.pvalue, 3))
OUT['per_target'] = rows
OUT['per_target_spearman'] = [dict(alpha=a, metric=c, rho=float(spearmanr(pt[c], pt[f'gain_{a}']).correlation), p=float(spearmanr(pt[c], pt[f'gain_{a}']).pvalue)) for a in S.LV for c in ['open_eq_high_or_low', 'corr_oc_prior_us']]

# ---------- regions ----------
REG = {'Asia': ['JPN', 'HKG', 'KOR', 'TWN', 'IND', 'CHN'], 'Europe': ['DEU', 'FRA', 'ESP', 'CHE', 'NLD', 'BEL', 'ITA'], 'Canada': ['CAN'], 'Southeast Asia': ['THA', 'MYS', 'PHL']}
rows = []
for a in S.LV:
    for rn, tg in REG.items():
        Tt = stack(mkey('oc', a, 'Tracker'), 'tk', 'oc', tg); Tb = stack(mkey('oc', a, 'LinBuf'), 'tk', 'oc', tg); est, l, h = ratio_ci(Tb - Tt, None, 'oc'); mt = np.nanmean(Tt)
        rows.append(dict(alpha=a, region=rn, pct=100 * est / mt, lo=100 * l / mt, hi=100 * h / mt))
OUT['regions'] = rows; print(pd.DataFrame(rows).round(2).to_string(index=False))
json.dump(OUT, open('/home/nafisnaufal1426/aciids/astar/oc4/analysis.json', 'w'), indent=1, default=float)
