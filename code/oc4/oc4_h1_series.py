"""v7 (round-5 R4): freeze the 2015-2025 pooled daily tick-loss differences (buffer minus tracker, mean over the 34 cells) used
to calibrate the H1 test under the null (series centred at zero) and to simulate its power. Writes h1_series.npz."""
import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/oc4')
import pickle
import numpy as np
import oc4_study as S
import oc4_analysis_lite as A
R = pickle.load(open('/home/nafisnaufal1426/aciids/astar/oc4/cells.pkl', 'rb'))
DD = {a: np.nanmean(A.daily(R, a, 'LinBuf') - A.daily(R, a, 'Tracker'), 1) for a in S.LV}
ok = np.isfinite(DD[0.01]) & np.isfinite(DD[0.05])
np.savez('/home/nafisnaufal1426/aciids/astar/oc4/h1_series.npz', d01=DD[0.01][ok], d05=DD[0.05][ok])
print(int(ok.sum()), DD[0.01][ok].mean(), DD[0.05][ok].mean())
