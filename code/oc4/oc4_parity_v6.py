"""v6 EXPLORATORY (round-4 R5, R11, S23): oc4_parity.py with grids extended until selections are interior where possible,
plus test-window backtests (Kupiec, Christoffersen independence, Basel yellow zone at 1%) and day-on-day VaR jumps for the
tuned baselines. Grids:
  QR    - window in {500, 1000, 2000, 3000, 4000, expanding}                                             6
  PID   - QR window x eta in {0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}                              48
  DtACI - QR window x residual window {500, 1000, 2000} x I {50, 100, 200, 400, 800}                      90
Raw series begin in January 2000; the first QR fits are in 2001-2002 (Italy 2004), so at the 2007-2014 dev window
'expanding' means about 1,300-3,500 days of history. Writes parity_v6.json.
Original docstring of oc4_parity.py (v5 EXPLORATORY, round-3 S9): tuning parity for the same-feature baselines. In v4 the quantile regression (QR) and DtACI had one
configuration each, PID six and the buffer sixteen. Here each baseline gets a small grid, tuned on 2007-2014 with the same rule
as oc4_study.py (min tick loss s.t. |viol - alpha| <= 0.1 alpha, pooled over targets and base models; else the closest rate):
  QR    - rolling window in {500, 1000, 2000} days (refit every 21 days)                         3 configurations
  PID   - QR window x proportional step eta in {0.01, 0.02, 0.05, 0.1, 0.2, 0.5}                 18
  DtACI - QR window x residual window in {500, 1000} x interval length I in {50, 100, 200}      18
Then test-window (2015-2025) tick loss and paired buffer-minus-baseline intervals (21-day block bootstrap). Open timing only.
Writes parity.json."""
import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/oc4')
import itertools, json, pickle
from multiprocessing import Pool
import numpy as np, pandas as pd
import oc4_lib as L
import oc4_study as S
D = S.D
WINS = [500, 1000, 2000, 3000, 4000, 1000000]; ETAS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]; RWINS = [500, 1000, 2000]; IS = [50, 100, 200, 400, 800]


def shat_job(t):
    k, b, a, win = t; s, v, ok = S.scores(k, b, a, 'oc'); i0 = int(np.argmax(ok))
    sh = np.full(len(s), np.nan); sh[i0:] = L.qr_scorecaster(s[i0:], D[k]['oc']['F'][i0:], a, S.BURN, window=win)
    return t, sh


def evaluate(k, b, a, sh, meth, cfg):
    s, v, ok = S.scores(k, b, a, 'oc'); i0 = int(np.argmax(ok)); ss = s[i0:]; hh = sh[i0:]
    if meth == 'QR': q = hh.copy(); q[:S.BURN] = np.nan
    elif meth == 'PID': q = L.pid_p(ss, hh, a, cfg['eta'], S.BURN)
    else: q = L.dtaci(ss, hh, a, S.BURN, cfg['rwin'], S.GAMMAS, cfg['I'])
    Q = np.full(len(s), np.nan); Q[i0:] = q; vv = v + D[k]['oc']['sig'] * Q; Lk = D[k]['oc']['loss']
    e = np.where(np.isfinite(vv), (Lk > vv).astype(float), np.nan); u = vv - Lk; tk = (a - (u < 0)) * u
    return e, tk, vv


def cfgs():
    out = [('QR', dict(win=w)) for w in WINS]
    out += [('PID', dict(win=w, eta=e)) for w, e in itertools.product(WINS, ETAS)]
    out += [('DtACI', dict(win=w, rwin=rw, I=i)) for w, rw, i in itertools.product(WINS, RWINS, IS)]
    return out


SH = None
def run_job(t):
    k, b, a, meth, cj = t; cfg = json.loads(cj); e, tk, vv = evaluate(k, b, a, SH[(k, b, a, cfg['win'])], meth, cfg)
    return t, (e.astype(np.float32), tk.astype(np.float32), vv.astype(np.float32))


def init(sh):
    global SH; SH = sh


if __name__ == '__main__':
    with Pool(204) as p:
        SHd = dict(p.map(shat_job, [(k, b, a, w) for k in S.TG for b in S.BASES for a in S.LV for w in WINS]))
    tasks = [(k, b, a, m, json.dumps(c)) for k in S.TG for b in S.BASES for a in S.LV for m, c in cfgs()]
    with Pool(200, initializer=init, initargs=(SHd,)) as p:
        RR = dict(p.map(run_job, tasks, chunksize=4))
    # dev selection
    rows = []
    for a in S.LV:
        for m, c in cfgs():
            cj = json.dumps(c); tks, vs = [], []
            for k in S.TG:
                for b in S.BASES:
                    e, tk, _ = RR[(k, b, a, m, cj)]; x, y = S.score_window(k, e, tk, *L.DEV, 'oc'); tks.append(x); vs.append(y)
            rows.append(dict(alpha=a, meth=m, cfg=cj, tick=float(np.mean(tks)), viol=float(np.mean(vs))))
    dev = pd.DataFrame(rows); sel = {}; EDGE = {}
    for (a, m), x in dev.groupby(['alpha', 'meth']):
        ok = x[np.abs(x.viol - a) <= 0.1 * a]; best = (ok.sort_values('tick') if len(ok) else x.assign(dd=np.abs(x.viol - a)).sort_values('dd')).iloc[0]
        sel[(a, m)] = best.cfg
        c_ = json.loads(best.cfg); edges = {'win': WINS, 'eta': ETAS, 'rwin': RWINS, 'I': IS}
        at_edge = {kk: (vv_ in (edges[kk][0], edges[kk][-1])) for kk, vv_ in c_.items() if kk in edges}
        EDGE[(a, m)] = at_edge; print('edge', a, m, at_edge, flush=True); print('dev', a, m, best.cfg, 'tick', round(100 * best.tick, 3), 'viol', round(100 * best.viol, 2), 'rule_met', len(ok) > 0, flush=True)
    # test comparison with the v4 buffer and tracker
    R4 = pickle.load(open('/home/nafisnaufal1426/aciids/astar/oc4/cells.pkl', 'rb'))
    lo, hi = L.TEST; U = np.unique(np.concatenate([D[k]['oc']['dates'] for k in S.TG])); M = (U >= np.datetime64(lo)) & (U < np.datetime64(hi))
    def og(k, x): o = np.full(len(U), np.nan); o[np.searchsorted(U, D[k]['oc']['dates'])] = x; return o[M]
    def st(fn): return np.column_stack([og(k, fn(k, b)) for k in S.TG for b in S.BASES])
    rng = np.random.default_rng(13); Tn = int(M.sum()); stt = rng.integers(0, Tn, (3000, int(np.ceil(Tn / 21)))); I = ((stt[:, :, None] + np.arange(21)).reshape(3000, -1)[:, :Tn]) % Tn
    def ci(X):
        s_ = np.nansum(X, 1); c = np.sum(np.isfinite(X), 1).astype(float); bb = s_[I].sum(1) / c[I].sum(1); return s_.sum() / c.sum(), np.percentile(bb, 2.5), np.percentile(bb, 97.5)
    out = []
    for a in S.LV:
        Tt = st(lambda k, b: R4[('oc', k, b, a, 'Tracker')]['tk']); mt = np.nanmean(Tt); Tb = st(lambda k, b: R4[('oc', k, b, a, 'LinBuf')]['tk'])
        for m in ['QR', 'PID', 'DtACI']:
            Tm = st(lambda k, b: RR[(k, b, a, m, sel[(a, m)])][1]); Em = st(lambda k, b: RR[(k, b, a, m, sel[(a, m)])][0])
            e1, l1, h1 = ci(Tm - Tt); e2, l2, h2 = ci(Tb - Tm)
            out.append(dict(alpha=a, method=m, cfg=sel[(a, m)], viol=100 * float(np.nanmean(Em)), vs_tracker=100 * e1 / mt, vs_tracker_lo=100 * l1 / mt, vs_tracker_hi=100 * h1 / mt,
                            buffer_minus=100 * e2 / mt, buffer_minus_lo=100 * l2 / mt, buffer_minus_hi=100 * h2 / mt))
    print(pd.DataFrame(out).round(2).to_string(index=False))
    from scipy.stats import chi2
    def kupiec(x, a):
        n = len(x); k = x.sum(); ph = k / n
        if k in (0, n): return 2 * (-(n - k) * np.log(1 - a) - k * np.log(a) + (n * np.log(1 - ph) if k == 0 else n * np.log(ph)))
        return -2 * ((n - k) * np.log(1 - a) + k * np.log(a) - (n - k) * np.log(1 - ph) - k * np.log(ph))
    def chr_ind(x):
        x0, x1 = x[:-1], x[1:]; n00 = np.sum((x0 == 0) & (x1 == 0)); n01 = np.sum((x0 == 0) & (x1 == 1)); n10 = np.sum((x0 == 1) & (x1 == 0)); n11 = np.sum((x0 == 1) & (x1 == 1))
        p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1); p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
        def ll(p, a, b): return (a * np.log(1 - p) if a else 0) + (b * np.log(p) if b else 0) if 0 < p < 1 else 0.0
        return -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    def cell_series(k, b, a, m):
        if m in ('Tracker', 'LinBuf'): d = R4[('oc', k, b, a, m)]; return og(k, d['e']), og(k, d['vv'])
        e, _, vv = RR[(k, b, a, m, sel[(a, m)])]; return og(k, e), og(k, vv)
    bt = []
    for a in S.LV:
        for m in ['Tracker', 'LinBuf', 'QR', 'PID', 'DtACI']:
            kr = ir = n = 0; yel = red = nw = 0; jumps = []; ev = []
            for k in S.TG:
                for b in S.BASES:
                    e, vv = cell_series(k, b, a, m); ok = np.isfinite(e); x = e[ok].astype(int); n += 1; ev.append(x.mean())
                    kr += kupiec(x, a) > chi2.ppf(0.95, 1); ir += chr_ind(x) > chi2.ppf(0.95, 1)
                    if a == 0.01:
                        w = np.convolve(x, np.ones(250, int), 'valid'); nw += len(w); yel += np.sum((w >= 5) & (w <= 9)); red += np.sum(w >= 10)
                    V = vv[ok]; J = np.diff(V) / np.abs(V[:-1]); jumps.append(J[np.isfinite(J)])
            J = np.concatenate(jumps)
            bt.append(dict(alpha=a, method=m, viol=100 * float(np.mean(ev)), kupiec_rej=int(kr), ind_rej=int(ir), cells=n, yellow=(100 * yel / nw if nw else None), red=(100 * red / nw if nw else None),
                           share_rise_gt50=100 * float(np.mean(J > 0.5)), mean_abs_change=100 * float(np.mean(np.abs(J)))))
    print(pd.DataFrame(bt).round(2).to_string(index=False))
    json.dump(dict(selected={f'{a}|{m}': c for (a, m), c in sel.items()}, at_edge={f'{a}|{m}': v for (a, m), v in EDGE.items()}, dev=rows, test=out, backtests=bt),
              open('/home/nafisnaufal1426/aciids/astar/oc4/parity_v6.json', 'w'), indent=1, default=float)
