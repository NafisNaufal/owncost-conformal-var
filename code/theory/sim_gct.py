"""Gated conformal tracker (GCT) under repeated bursts. Synthetic only.
Calm: s ~ N(0,1); burst of m days: s ~ N(0, sig^2), every g days (burst at the end of each cycle).
Information stream X_t (known before forecast t): own = s_{t-1}^2; side = sum_j W_{j,t}^2, W abnormal from tau-ell to tau+m-1.
Detector: conformal ranks of X_t against history since the last restart; Shiryaev-Roberts mixture over power calibrators
f_th(u) = (1-th)(1-u)^(-th) (distribution-independent dictionary of nondecreasing densities); alarm when statistic >= gamma; hold state 1 for r days, then restart.
Thresholds: two quantile trackers, one per state. Laws, lead and burst size are not given to the rule.
Measured: calm cost (exact normal pinball excess) on calm days outside recovery, violations per burst, alarm-day coverage."""
import sys, json
import numpy as np
from numba import njit
from scipy.stats import norm
from multiprocessing import Pool

KAPPAS = np.array([0.3, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95])  # tail calibrators f(u) = (1-th)(1-u)^(-th), nondecreasing densities on [0,1]


@njit(cache=True)
def run_path(seed, a, sig, m, g, ell, mode, Kside, gamma, r, eta0, eta1, ncyc, kap, use_det, oracle):
    np.random.seed(seed)
    q0 = 0.0; q1 = 0.0
    # initialise both trackers at the calm quantile
    # (normal quantile passed through via a precomputed constant would be cleaner; use a burn-in estimate)
    hist = np.empty(20000); nh = 0
    R = np.zeros(len(kap)); pi = np.ones(len(kap)) / len(kap)
    hold = 0; restart_pending = False
    sprev = 0.0
    T = g * ncyc
    calm_h = 0.0; calm_n = 0; burst_v = np.zeros(ncyc); alarm_e = 0.0; alarm_n = 0; rec_h = 0.0; rec_n = 0
    day0 = np.zeros(ncyc); alarm_day = np.full(ncyc, -1)
    first = True
    qa = 0.0
    for t in range(T):
        cyc = t // g; pos = t % g; tau_pos = g - m
        inburst = pos >= tau_pos
        side_ab = pos >= tau_pos - ell
        # information available before forecast t
        if mode == 0:
            x = sprev * sprev
        else:
            x = 0.0
            for j in range(Kside):
                w = np.random.randn() * (sig if side_ab else 1.0)
                x += w * w
        # detector
        alarm_state = hold > 0
        if use_det and not alarm_state:
            # conformal rank of x among history (randomised ties are measure-zero here)
            S = 0.0
            if oracle:
                kk = 1.0 if mode == 0 else Kside * 1.0
                lr = sig ** (-kk) * np.exp(0.5 * x * (1.0 - 1.0 / sig ** 2))
                R[0] = lr * (R[0] + 1.0); S = R[0]
            else:
                cnt = 0
                for i in range(nh):
                    if hist[i] < x:
                        cnt += 1
                u = (cnt + np.random.rand()) / (nh + 1.0)
                for j in range(len(kap)):
                    R[j] = (1.0 - kap[j]) * (1.0 - u) ** (-kap[j]) * (R[j] + 1.0)
                    S += pi[j] * R[j]
            if not oracle:
                hist[nh] = x; nh += 1
            if S >= gamma:
                hold = r; alarm_state = True
                if t >= 2 * g and alarm_day[cyc] < 0 and pos >= tau_pos - ell:
                    alarm_day[cyc] = pos - tau_pos
        if first:
            first = False
        thr = q1 if alarm_state else q0
        s = np.random.randn() * (sig if inburst else 1.0)
        e = 1.0 if s > thr else 0.0
        # bookkeeping after burn-in (two cycles)
        if t >= 2 * g:
            if inburst:
                burst_v[cyc] += e
                if pos == tau_pos:
                    day0[cyc] = e
            else:
                # exact calm excess pinball of the threshold used
                z = thr
                pdf = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
                sf = 0.5 * erfc(z / np.sqrt(2.0))
                es = pdf - z * sf
                lb = (1 - a) * es + a * (z + es)
                if pos < tau_pos - 30 and pos > 30:
                    calm_h += lb; calm_n += 1
                elif pos <= 30:
                    rec_h += lb; rec_n += 1
            if alarm_state:
                alarm_e += e; alarm_n += 1
        # tracker updates, per state
        if alarm_state:
            q1 += eta1 * (e - a)
        else:
            q0 += eta0 * (e - a)
        sprev = s
        if hold > 0:
            hold -= 1
            if hold == 0:
                nh = 0; R[:] = 0.0  # restart the detector with a fresh history
    return calm_h, calm_n, burst_v[2:], day0[2:], alarm_e, alarm_n, rec_h, rec_n, alarm_day[2:]


@njit(cache=True)
def erfc(x):
    # Abramowitz-Stegun 7.1.26 complement, adequate for cost accounting
    z = abs(x); t = 1.0 / (1.0 + 0.5 * z)
    r = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return r if x >= 0 else 2.0 - r


def job(args):
    seed, cfg = args
    a, sig, m, g, ell, mode, K, gamma, r, eta0, eta1, ncyc, use_det, oracle = cfg
    out = run_path(seed, a, sig, m, g, ell, mode, K, gamma, r, eta0, eta1, ncyc, KAPPAS, use_det, oracle)
    return out


def summarise(res, a):
    q0 = norm.ppf(1 - a); pdf, sf = norm.pdf(q0), norm.sf(q0); es = pdf - q0 * sf; l0 = (1 - a) * es + a * (q0 + es)
    ch = sum(r[0] for r in res); cn = sum(r[1] for r in res)
    bv = np.concatenate([r[2] for r in res]); d0 = np.concatenate([r[3] for r in res]); ae = sum(r[4] for r in res); an = sum(r[5] for r in res)
    rh = sum(r[6] for r in res); rn = sum(r[7] for r in res); ad = np.concatenate([r[8] for r in res])
    return dict(delta=ch / cn - l0, V=float(bv.mean()), V_se=float(bv.std() / np.sqrt(len(bv))), d0=float(d0.mean()), alarm_cov=ae / max(an, 1), alarm_share=an / (len(res) * 1.0), rec_cost=rh / max(rn, 1) - l0, detected=float(np.mean(ad >= 0)), delay_med=float(np.median(ad[ad >= 0])) if (ad >= 0).any() else None)


if __name__ == '__main__':
    a = 0.05; rows = []
    with Pool(120) as p:
        for sig, m in [(2.0, 10), (2.0, 60), (3.0, 10), (3.0, 40)]:
            for oracle in [True, False]:
                for gamma in [30, 300, 3000, 30000, 300000]:
                    cfg = (a, sig, m, 500, 0, 0, 0, float(gamma), max(m + 10, 20), 0.02, 0.5, 18, True, oracle)
                    res = p.map(job, [(s_, cfg) for s_ in range(200)])
                    d = summarise(res, a); d.update(sig=sig, m=m, rule='oracle SR (own, known laws)' if oracle else 'conformal SR (own, laws unknown)', gamma=gamma); rows.append(d)
                    print(f"sig={sig} m={m} {'oracle' if oracle else 'conformal'} log_gamma={np.log(gamma):5.1f} delta={d['delta']:.5f} V={d['V']:.2f} detected={d['detected']:.2f} delay_med={d['delay_med']} alarm_cov={d['alarm_cov']:.3f}", flush=True)
    json.dump(rows, open('/home/nafisnaufal1426/aciids/astar/theory/sim_gct_scaling_tail.json', 'w'), indent=1)
