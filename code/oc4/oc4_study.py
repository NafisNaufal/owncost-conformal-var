"""v4 EXPLORATORY study: tune every method on 2007-2014 (min tick loss s.t. |viol - alpha| <= 0.1 alpha, pooled over the 17
targets and two base models), for both timings. Evaluation is in oc4_eval.py."""
import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/oc4')
import itertools, json, pickle
from multiprocessing import Pool
import numpy as np, pandas as pd
import oc4_lib as L
D = pickle.load(open('/home/nafisnaufal1426/aciids/astar/oc4/oc4_data.pkl', 'rb'))
TG = L.FRESH_T + L.SEA_T; BASES = ['EWMA', 'HS']; LV = [0.01, 0.05]; TH = np.array([0.3, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95]); BURN = 250
GAMMAS = np.array([0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128])


def streams(k, tm='oc', shift=0, fmask=None, shift_cols=None):
    """Information streams. shift > 0 rolls the cross-market columns (all except the own gap) by `shift` rows."""
    r = D[k][tm]; F = r['F'].copy(); us2 = r['us2'].copy(); glob = r['glob'].copy()
    if shift:
        cols = shift_cols if shift_cols is not None else (list(range(1, F.shape[1])) if tm == 'oc' else list(range(F.shape[1])))
        F[:, cols] = np.roll(F[:, cols], shift, axis=0); us2 = np.roll(us2, shift); glob = np.roll(glob, shift)
    if fmask is not None: F = F * np.asarray(fmask)[None, :]
    out = dict(F=F, us=us2, glob=glob)
    if tm == 'oc': out['own'] = r['own2']; out['comb'] = r['own2'] + glob
    return out


def scores(k, base, a, tm='oc'):
    r = D[k][tm]; v = r[('vhat', base, a)]; s = (r['loss'] - v) / r['sig']; ok = np.isfinite(s)
    return s, v, ok


def run(k, base, a, meth, cfg, tm='oc', shift=0, seed=0, fmask=None, shift_cols=None):
    s, v, ok = scores(k, base, a, tm); i0 = int(np.argmax(ok)); ss = s[i0:]; X = streams(k, tm, shift, fmask, shift_cols); st = np.zeros(len(ss))
    if meth == 'Tracker': q = L.tracker(ss, a, cfg['eta'], BURN)
    elif meth.startswith('GCT'): q, st = L.gct(ss, X[meth.split('-')[1]][i0:], a, cfg['gamma'], cfg['eta0'], cfg['eta1'], cfg['r'], BURN, TH, seed)
    elif meth == 'LinBuf': q = L.linear_buffer(ss, X['F'][i0:], a, cfg['eta_u'], cfg['eta_w'], 6.0, BURN)
    elif meth == 'Base': q = np.zeros(len(ss)); q[:BURN] = np.nan
    elif meth == 'QR': q = D[k][tm][('shat', base, a)][i0:].copy(); q[:BURN] = np.nan
    elif meth == 'PID': q = L.pid_p(ss, D[k][tm][('shat', base, a)][i0:], a, cfg['eta'], BURN)
    elif meth == 'DtACI': q = L.dtaci(ss, D[k][tm][('shat', base, a)][i0:], a, BURN, 500, GAMMAS, 100)
    Q = np.full(len(s), np.nan); Q[i0:] = q; ST = np.zeros(len(s)); ST[i0:] = st
    vv = v + D[k][tm]['sig'] * Q; Lk = D[k][tm]['loss']; e = np.where(np.isfinite(vv), (Lk > vv).astype(float), np.nan); u = vv - Lk; tk = (a - (u < 0)) * u
    return e, tk, ST, vv


def score_window(k, e, tk, lo, hi, tm='oc'):
    d = D[k][tm]['dates']; m = (d >= np.datetime64(lo)) & (d < np.datetime64(hi)) & np.isfinite(e)
    return np.nanmean(tk[m]), np.nanmean(e[m])


def grid(tm):
    g = [('Tracker', dict(eta=x)) for x in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]] + [('Base', {})]
    g += [('LinBuf', dict(eta_u=u, eta_w=w)) for u, w in itertools.product([0.02, 0.05, 0.1, 0.2], [0.1, 0.5, 2.0, 5.0])]
    gst = ['own', 'us', 'glob', 'comb'] if tm == 'oc' else ['us']
    for st in gst:
        g += [(f'GCT-{st}', dict(gamma=G, eta0=e0, eta1=e1, r=r)) for G, e0, e1, r in itertools.product([10, 30, 100, 300, 1000], [0.02, 0.05, 0.1, 0.2], [0.1, 0.5], [3, 10, 20])]
    if tm == 'oc':
        g += [('QR', {}), ('DtACI', {})] + [('PID', dict(eta=x)) for x in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]]
    return g


def dev_job(t):
    tm, a, meth, cfg = t; tks, vs = [], []
    for k in TG:
        for b in BASES:
            e, tk, st, vv = run(k, b, a, meth, cfg, tm); x, y = score_window(k, e, tk, *L.DEV, tm); tks.append(x); vs.append(y)
    return dict(timing=tm, alpha=a, meth=meth, cfg=json.dumps(cfg), tick=float(np.mean(tks)), viol=float(np.mean(vs)))


def select(dev):
    sel = {}
    for (tm, a, m), x in dev.groupby(['timing', 'alpha', 'meth']):
        ok = x[np.abs(x.viol - a) <= 0.1 * a]; b = (ok if len(ok) else x.assign(dd=np.abs(x.viol - a))).sort_values('tick' if len(ok) else 'dd').iloc[0]
        sel[(tm, a, m)] = json.loads(b.cfg)
        g = [json.loads(c) for c in x.cfg]; edge = []
        for key in sel[(tm, a, m)]:
            vals = sorted({c[key] for c in g})
            if len(vals) > 1 and sel[(tm, a, m)][key] in (vals[0], vals[-1]): edge.append(key)
        print(f'dev {tm} a={a} {m:9s} tick={100 * b.tick:.3f} viol={100 * b.viol:.2f} cfg={b.cfg} rule_met={len(ok) > 0} n_grid={len(x)} edge={edge}')
    return sel


if __name__ == '__main__':
    tasks = [(tm, a, m, c) for tm in ['oc', 'cc'] for a in LV for m, c in grid(tm)]
    print('configurations:', len(tasks))
    with Pool(200) as p:
        dev = pd.DataFrame(p.map(dev_job, tasks, chunksize=1))
    dev.to_pickle('/home/nafisnaufal1426/aciids/astar/oc4/dev.pkl')
    sel = select(dev); pickle.dump(sel, open('/home/nafisnaufal1426/aciids/astar/oc4/selected.pkl', 'wb'))
