"""Frozen entry point for the confirmatory test (protocol amendment 2, round-4 S13, R7).
Usage:
  python oc4_confirm.py --start 2026-01-01 --end YYYY-MM-DD --new-data DIR --out results.json
  python oc4_confirm.py --start 2015-01-01 --end 2026-01-01 --new-data none --out check.json   (reproduces the exploratory cells)
Steps:
 1. Verify the 31 frozen raw series against data_checksums.sha256; abort on any mismatch.
 2. Vintage rule: rows dated on or before 2025-12-31 always come from the frozen files; rows dated 2026-01-01 or later come only
    from DIR/<market>.csv (a fresh download, same vendor and columns). History is never re-downloaded or replaced. Merged files
    are written to a new directory next to --out.
 3. Rebuild the inputs with the frozen oc4_build.build (VaR at the local open, open-to-close loss), and run the quantile tracker
    and the learned buffer with the frozen settings in selected.pkl. No re-tuning.
 4. H1 (primary): for each level, d_t = mean over the 34 target x base cells of (buffer tick loss - tracker tick loss) on day t of
    the window. Statistic: dbar = mean of d_t. One-sided p-value from a 21-day circular block bootstrap of the centred series,
    3000 draws, seed 2026: p = (1 + #{b : mean(d*_b) - dbar <= dbar}) / 3001. Holm over the two levels: let p(1) <= p(2);
    reject the first if p(1) <= 0.025 and dbar < 0, then the second if p(2) <= 0.05 and dbar < 0; otherwise stop.
 5. Descriptive only (amendment 2): first-day changes at onsets, the own-gap-only buffer contrast, backtests.
Writes the JSON named by --out."""
import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/oc4')
import argparse, hashlib, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path('/home/nafisnaufal1426/aciids/astar/oc4'); CUT = np.datetime64('2025-12-31')


def verify():
    bad = []
    for line in open(HERE / 'data_checksums.sha256'):
        h, p = line.split(None, 1); p = p.strip()
        if hashlib.sha256(open(p, 'rb').read()).hexdigest() != h: bad.append(p)
    if bad: raise SystemExit(f'frozen data changed: {bad}')


def h1_test(d, seed=2026, B=3000, blk=21):
    d = np.asarray(d, float); d = d[np.isfinite(d)]; T = len(d); dbar = d.mean(); c = d - dbar; rng = np.random.default_rng(seed)
    st = rng.integers(0, T, (B, int(np.ceil(T / blk)))); I = ((st[:, :, None] + np.arange(blk)).reshape(B, -1)[:, :T]) % T
    mb = c[I].mean(1); p = (1 + np.sum(mb <= dbar)) / (B + 1)
    return float(dbar), float(p), T


def holm(res):
    """res: {alpha: (dbar, p)} -> {alpha: reject}; one-sided, family-wise 5%."""
    order = sorted(res, key=lambda a: res[a][1]); out = {a: False for a in res}
    for j, a in enumerate(order):
        dbar, p = res[a]
        if dbar < 0 and p <= 0.05 / (len(order) - j): out[a] = True
        else: break
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--start', required=True); ap.add_argument('--end', required=True)
    ap.add_argument('--new-data', required=True); ap.add_argument('--out', required=True); A_ = ap.parse_args()
    verify()
    import oc4_lib as L
    if A_.new_data != 'none':
        mdir = Path(A_.out).resolve().parent / 'merged_inputs'; mdir.mkdir(exist_ok=False)
        for m, p in list(L.SRC.items()):
            old = pd.read_csv(p, parse_dates=['Date']); old = old[old.Date <= pd.Timestamp(str(CUT))]
            new = pd.read_csv(Path(A_.new_data) / f'{m}.csv', parse_dates=['Date']); new = new[new.Date > pd.Timestamp(str(CUT))]
            pd.concat([old, new[old.columns]]).to_csv(mdir / f'{m}.csv', index=False); L.SRC[m] = mdir / f'{m}.csv'
    L.TEST = (A_.start, A_.end)
    import oc4_build as Bd  # loads the (possibly merged) series at import
    from multiprocessing import Pool
    with Pool(len(Bd.TG)) as p:
        D = dict(p.map(Bd.build, Bd.TG))
    import oc4_study as S
    S.D = D; sel = pickle.load(open(HERE / 'selected.pkl', 'rb'))
    U = np.unique(np.concatenate([D[k]['oc']['dates'] for k in S.TG])); M = (U >= np.datetime64(A_.start)) & (U < np.datetime64(A_.end))
    def og(k, x): o = np.full(len(U), np.nan); o[np.searchsorted(U, D[k]['oc']['dates'])] = x; return o[M]
    out = dict(start=A_.start, end=A_.end, new_data=A_.new_data, levels={}); res = {}
    for a in S.LV:
        cols_t, cols_b = [], []
        for k in S.TG:
            for b in S.BASES:
                _, tkt, _, _ = S.run(k, b, a, 'Tracker', sel[('oc', a, 'Tracker')]); _, tkb, _, _ = S.run(k, b, a, 'LinBuf', sel[('oc', a, 'LinBuf')])
                cols_t.append(og(k, tkt)); cols_b.append(og(k, tkb))
        Tt = np.column_stack(cols_t); Tb = np.column_stack(cols_b); d = np.nanmean(Tb - Tt, 1)
        dbar, p, n = h1_test(d); res[a] = (dbar, p)
        out['levels'][str(a)] = dict(days=n, dbar=dbar, pct_of_tracker=100 * float(np.nansum(Tb - Tt) / np.sum(np.isfinite(Tb - Tt)) / np.nanmean(Tt)), p_one_sided=p)
    rej = holm(res)
    for a in S.LV: out['levels'][str(a)]['holm_reject'] = bool(rej[a])
    json.dump(out, open(A_.out, 'w'), indent=1); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
