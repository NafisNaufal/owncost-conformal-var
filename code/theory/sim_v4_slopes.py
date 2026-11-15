"""R11: delay slope vs log(gamma) with path-bootstrap error, and violations per burst at the same gamma, known laws vs a
fixed-score conformal Shiryaev-Roberts detector (tail-calibrator mixture; NOT the learned-score construction of Theorem 2)."""
import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/theory')
import json
import numpy as np
from multiprocessing import Pool
import sim_gct as G
A = 0.05; GAMMAS = [30, 300, 3000, 30000, 300000]; NP = 200
def job(args):
    seed, cfg = args; out = G.run_path(seed, *cfg[:12], G.KAPPAS, cfg[12], cfg[13]); return seed, np.asarray(out[2]), np.asarray(out[8])
def KL(s): return 0.5 * (s ** 2 - 1 - np.log(s ** 2))
if __name__ == '__main__':
    res = {}
    with Pool(200) as p:
        for sig, m in [(2.0, 60), (3.0, 40)]:
            for oracle in [True, False]:
                for g in GAMMAS:
                    cfg = (A, sig, m, 500, 0, 0, 0, float(g), max(m + 10, 20), 0.02, 0.5, 18, True, oracle)
                    res[(sig, m, oracle, g)] = p.map(job, [(s, cfg) for s in range(NP)])
    rng = np.random.default_rng(0); out = []
    for sig, m in [(2.0, 60), (3.0, 40)]:
        D = KL(sig); lg = np.log(GAMMAS)
        def slope(oracle, idx):
            med = []
            for g in GAMMAS:
                dl = np.concatenate([res[(sig, m, oracle, g)][i][2] for i in idx]); dl = dl[dl >= 0]; med.append(np.median(dl))
            return np.polyfit(lg, med, 1)[0]
        full = np.arange(NP); bo = []; bc = []; br = []
        for _ in range(500):
            idx = rng.integers(0, NP, NP); so, sc = slope(True, idx), slope(False, idx); bo.append(so); bc.append(sc); br.append(sc / so)
        so, sc = slope(True, full), slope(False, full)
        row = dict(sig=sig, m=m, inv_D=1 / D, slope_known=so, slope_known_se=float(np.std(bo)), slope_df=sc, slope_df_se=float(np.std(bc)),
                   ratio=sc / so, ratio_lo=float(np.percentile(br, 2.5)), ratio_hi=float(np.percentile(br, 97.5)), ratio_vs_invD=sc * D)
        for g in [300, 3000, 30000]:
            vk = np.concatenate([res[(sig, m, True, g)][i][1] for i in full]); vd = np.concatenate([res[(sig, m, False, g)][i][1] for i in full])
            bb = []
            for _ in range(500):
                idx = rng.integers(0, NP, NP); a_ = np.concatenate([res[(sig, m, True, g)][i][1] for i in idx]).mean(); b_ = np.concatenate([res[(sig, m, False, g)][i][1] for i in idx]).mean()
                bb.append((b_ - A * m) / (a_ - A * m))
            row[f'V_known_g{g}'] = float(vk.mean()); row[f'V_df_g{g}'] = float(vd.mean()); row[f'excess_ratio_g{g}'] = float((vd.mean() - A * m) / (vk.mean() - A * m))
            row[f'excess_ratio_g{g}_lo'] = float(np.percentile(bb, 2.5)); row[f'excess_ratio_g{g}_hi'] = float(np.percentile(bb, 97.5))
        out.append(row); print(json.dumps(row, indent=0, default=float))
    json.dump(out, open('/home/nafisnaufal1426/aciids/astar/theory/sim_v4_slopes.json', 'w'), indent=1, default=float)
