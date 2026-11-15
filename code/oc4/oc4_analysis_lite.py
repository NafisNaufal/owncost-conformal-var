import sys; sys.path.insert(0, '/home/nafisnaufal1426/aciids/astar/oc4')
import pickle, numpy as np, oc4_lib as L, oc4_study as S
D = S.D
def daily(R, a, m, tm='oc'):
    U = np.unique(np.concatenate([D[k][tm]['dates'] for k in S.TG])); msk = (U >= np.datetime64(L.TEST[0])) & (U < np.datetime64(L.TEST[1]))
    cols = []
    for k in S.TG:
        for b in S.BASES:
            x = np.full(len(U), np.nan); x[np.searchsorted(U, D[k][tm]['dates'])] = R[(tm, k, b, a, m)]['tk']; cols.append(x[msk])
    return np.column_stack(cols)
