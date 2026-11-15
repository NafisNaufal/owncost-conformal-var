"""v7 EXPLORATORY (round-5 S10, S11): sensitivity of the tuned-baseline comparison of oc4_parity_v6.py.
On 2007-2014 the 4000-day and expanding windows give identical development results (no cell has more than about 3,830 days
of history), so v6's choice between them fell to sort order. Here every tied pair is evaluated on the test window, together
with the lowest-development-loss configurations of QR and DtACI at 1% (which meet the violation-rate rule for no
configuration and were selected in v6 by closest violation rate). Paired buffer-minus-baseline intervals as in v6.
Writes parity_v7.json."""
import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/oc4')
import json, pickle
from multiprocessing import Pool
import numpy as np
import oc4_lib as L
import oc4_study as S
import oc4_parity_v6 as P
D = S.D
CF = [(0.01, 'QR', dict(win=2000)), (0.01, 'QR', dict(win=4000)), (0.01, 'QR', dict(win=1000000)),
      (0.01, 'PID', dict(win=1000000, eta=0.005)), (0.01, 'PID', dict(win=4000, eta=0.005)),
      (0.01, 'DtACI', dict(win=500, rwin=500, I=800)), (0.01, 'DtACI', dict(win=1000000, rwin=2000, I=800)), (0.01, 'DtACI', dict(win=4000, rwin=2000, I=800)),
      (0.05, 'QR', dict(win=4000)), (0.05, 'QR', dict(win=1000000)),
      (0.05, 'PID', dict(win=1000000, eta=0.002)), (0.05, 'PID', dict(win=4000, eta=0.002)),
      (0.05, 'DtACI', dict(win=4000, rwin=1000, I=800)), (0.05, 'DtACI', dict(win=1000000, rwin=1000, I=800))]
SH = None
def init(sh):
    global SH; SH = sh
def run_job(t):
    k, b, a, m, cj = t; c = json.loads(cj); e, tk, vv = P.evaluate(k, b, a, SH[(k, b, a, c['win'])], m, c); return t, tk.astype(np.float32)
if __name__ == '__main__':
    wins = sorted({c['win'] for _, _, c in CF})
    with Pool(150) as p:
        SHd = dict(p.map(P.shat_job, [(k, b, a, w) for k in S.TG for b in S.BASES for a in S.LV for w in wins]))
    tasks = [(k, b, a, m, json.dumps(c)) for a, m, c in CF for k in S.TG for b in S.BASES]
    with Pool(150, initializer=init, initargs=(SHd,)) as p:
        RR = dict(p.map(run_job, tasks, chunksize=2))
    R4 = pickle.load(open('/home/nafisnaufal1426/aciids/astar/oc4/cells.pkl', 'rb'))
    lo, hi = L.TEST; U = np.unique(np.concatenate([D[k]['oc']['dates'] for k in S.TG])); M = (U >= np.datetime64(lo)) & (U < np.datetime64(hi))
    def og(k, x): o = np.full(len(U), np.nan); o[np.searchsorted(U, D[k]['oc']['dates'])] = x; return o[M]
    def st(fn): return np.column_stack([og(k, fn(k, b)) for k in S.TG for b in S.BASES])
    rng = np.random.default_rng(13); Tn = int(M.sum()); stt = rng.integers(0, Tn, (3000, int(np.ceil(Tn / 21)))); I = ((stt[:, :, None] + np.arange(21)).reshape(3000, -1)[:, :Tn]) % Tn
    def ci(X):
        s_ = np.nansum(X, 1); c = np.sum(np.isfinite(X), 1).astype(float); bb = s_[I].sum(1) / c[I].sum(1); return s_.sum() / c.sum(), np.percentile(bb, 2.5), np.percentile(bb, 97.5)
    out = []
    for a, m, c in CF:
        cj = json.dumps(c); Tt = st(lambda k, b: R4[('oc', k, b, a, 'Tracker')]['tk']); mt = np.nanmean(Tt); Tb = st(lambda k, b: R4[('oc', k, b, a, 'LinBuf')]['tk'])
        Tm = st(lambda k, b: RR[(k, b, a, m, cj)]); e, l, h = ci(Tb - Tm)
        out.append(dict(alpha=a, method=m, cfg=c, buffer_minus=100 * e / mt, lo=100 * l / mt, hi=100 * h / mt))
        print(a, m, c, round(100 * e / mt, 2), [round(100 * l / mt, 2), round(100 * h / mt, 2)], flush=True)
    json.dump(out, open('/home/nafisnaufal1426/aciids/astar/oc4/parity_v7.json', 'w'), indent=1)
