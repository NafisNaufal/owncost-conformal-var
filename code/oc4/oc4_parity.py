"""v5 EXPLORATORY (round-3 S9): tuning parity for the same-feature baselines. In v4 the quantile regression (QR) and DtACI had one
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
WINS = [500, 1000, 2000]; ETAS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]; RWINS = [500, 1000]; IS = [50, 100, 200]


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
    return e, tk


def cfgs():
    out = [('QR', dict(win=w)) for w in WINS]
    out += [('PID', dict(win=w, eta=e)) for w, e in itertools.product(WINS, ETAS)]
    out += [('DtACI', dict(win=w, rwin=rw, I=i)) for w, rw, i in itertools.product(WINS, RWINS, IS)]
    return out


SH = None
def run_job(t):
    k, b, a, meth, cj = t; cfg = json.loads(cj); e, tk = evaluate(k, b, a, SH[(k, b, a, cfg['win'])], meth, cfg)
    return t, (e.astype(np.float32), tk.astype(np.float32))


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
                    e, tk = RR[(k, b, a, m, cj)]; x, y = S.score_window(k, e, tk, *L.DEV, 'oc'); tks.append(x); vs.append(y)
            rows.append(dict(alpha=a, meth=m, cfg=cj, tick=float(np.mean(tks)), viol=float(np.mean(vs))))
    dev = pd.DataFrame(rows); sel = {}
    for (a, m), x in dev.groupby(['alpha', 'meth']):
        ok = x[np.abs(x.viol - a) <= 0.1 * a]; best = (ok.sort_values('tick') if len(ok) else x.assign(dd=np.abs(x.viol - a)).sort_values('dd')).iloc[0]
        sel[(a, m)] = best.cfg; print('dev', a, m, best.cfg, 'tick', round(100 * best.tick, 3), 'viol', round(100 * best.viol, 2), 'rule_met', len(ok) > 0, flush=True)
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
    json.dump(dict(selected={f'{a}|{m}': c for (a, m), c in sel.items()}, dev=rows, test=out), open('/home/nafisnaufal1426/aciids/astar/oc4/parity.json', 'w'), indent=1, default=float)
