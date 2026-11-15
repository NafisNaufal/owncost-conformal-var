"""v4 library (EXPLORATORY study). Changes from v3 (astar/oc/oc_lib.py):
- session clocks are time-zone aware (zoneinfo, daylight saving included). A source session counts as known only after a
  conservative late close (closing auction included); the target reference time is the official open (open-to-close timing)
  or the official close of the previous session (previous-close timing);
- one stale-age rule (age <= 3 calendar days) for every cross-market input;
- baselines that see the same features: rolling linear quantile regression (scorecaster), conformal PID with that
  scorecaster, and DtACI on the scorecaster's residuals."""
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
ROOT = Path('/home/nafisnaufal1426/aciids')
SRC = {**{m: ROOT / 'paper/data/raw' / f'{m}.csv' for m in ['IDX', 'MYS', 'SGP', 'USA']}, **{m: ROOT / 'paper/data/manual' / f'{m}.csv' for m in ['THA', 'PHL']}}
for p in sorted((ROOT / 'astar/data/raw').glob('*.csv')):
    SRC.setdefault(p.stem, p)
VENDOR = {m: ('Stooq' if m in ('THA', 'PHL') else 'Yahoo Finance') for m in SRC}
PANEL = ['IDX', 'MYS', 'PHL', 'SGP', 'THA', 'USA', 'JPN', 'HKG', 'KOR', 'TWN', 'AUS', 'IND', 'CHN', 'GBR', 'DEU', 'FRA', 'ESP', 'CHE', 'NLD', 'BEL', 'AUT', 'ITA', 'TUR', 'CAN', 'BRA', 'MEX', 'ARG']
# (time zone, official open, official close, conservative late close incl. closing auction), local hours
SESS = {'JPN': ('Asia/Tokyo', 9, 15, 15.5), 'HKG': ('Asia/Hong_Kong', 9.5, 16, 16 + 10 / 60), 'KOR': ('Asia/Seoul', 9, 15.5, 15.5),
        'TWN': ('Asia/Taipei', 9, 13.5, 13.5), 'AUS': ('Australia/Sydney', 10, 16, 16 + 12 / 60), 'IND': ('Asia/Kolkata', 9.25, 15.5, 15.5),
        'CHN': ('Asia/Shanghai', 9.5, 15, 15), 'IDX': ('Asia/Jakarta', 9, 16, 16.25), 'MYS': ('Asia/Kuala_Lumpur', 9, 17, 17),
        'PHL': ('Asia/Manila', 9.5, 15.5, 15.75), 'SGP': ('Asia/Singapore', 9, 17, 17 + 16 / 60), 'THA': ('Asia/Bangkok', 10, 16.5, 16 + 40 / 60),
        'GBR': ('Europe/London', 8, 16.5, 16 + 35 / 60), 'TUR': ('Europe/Istanbul', 10, 18, 18 + 10 / 60),
        **{m: (tz, 9, 17.5, 17.75) for m, tz in [('DEU', 'Europe/Berlin'), ('FRA', 'Europe/Paris'), ('ESP', 'Europe/Madrid'), ('CHE', 'Europe/Zurich'),
                                              ('NLD', 'Europe/Amsterdam'), ('BEL', 'Europe/Brussels'), ('AUT', 'Europe/Vienna'), ('ITA', 'Europe/Rome')]},
        'CAN': ('America/Toronto', 9.5, 16, 16 + 10 / 60), 'USA': ('America/New_York', 9.5, 16, 16 + 10 / 60),
        'BRA': ('America/Sao_Paulo', 10, 17, 18), 'MEX': ('America/Mexico_City', 8.5, 15, 15 + 10 / 60), 'ARG': ('America/Argentina/Buenos_Aires', 11, 17, 17 + 10 / 60)}
FRESH_T = ['JPN', 'HKG', 'KOR', 'TWN', 'IND', 'CHN', 'DEU', 'FRA', 'ESP', 'CHE', 'NLD', 'BEL', 'ITA', 'CAN']
SEA_T = ['THA', 'MYS', 'PHL']
DEV = ('2007-01-01', '2015-01-01'); TEST = ('2015-01-01', '2026-01-01'); MAXAGE = 3.0


def utc_at(m, dates, which):
    """UTC time (datetime64[m]) of market m's session event on each local date; which in {'open', 'close', 'late'}."""
    tz, o, c, cl = SESS[m]; h = {'open': o, 'close': c, 'late': cl}[which]
    idx = pd.DatetimeIndex(dates) + pd.to_timedelta(round(h * 60), unit='m')
    loc = idx.tz_localize(tz, nonexistent='shift_forward', ambiguous=True)
    return loc.tz_convert('UTC').tz_localize(None).values.astype('datetime64[m]')


def ewma_sigma(r, lam=0.94, init=250):
    s2 = np.empty(len(r)); s2[0] = np.var(r[:init])
    for t in range(1, len(r)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2
    return np.sqrt(s2)


def load_ohlc(m):
    d = pd.read_csv(SRC[m], parse_dates=['Date']).set_index('Date').sort_index(); d = d[~d.index.duplicated(keep='last')]
    d = d[(d.Close > 0) & d.Close.notna()]
    out = pd.DataFrame(index=d.index); out['close'] = d.Close; out['r'] = 100 * np.log(d.Close).diff()
    out['open'] = d.Open.where(d.Open > 0); out['oc'] = 100 * np.log(d.Close / out.open); out['gap'] = 100 * np.log(out.open / d.Close.shift(1))
    out['high'] = d.High.where(d.High > 0) if 'High' in d else np.nan; out['low'] = d.Low.where(d.Low > 0) if 'Low' in d else np.nan
    out = out.iloc[1:]; out['sig_cc'] = ewma_sigma(out.r.fillna(0).to_numpy())
    return out


def side_info(target, data, ref_utc, ref_dates):
    """For each reference time: (z, exceedance flag, age in days) of each other market's latest session whose
    conservative late close precedes the reference time."""
    Z = {}
    for j in PANEL:
        if j == target: continue
        dj = data[j]; ts_close = utc_at(j, dj.index.values, 'late')
        i = np.searchsorted(ts_close, ref_utc, side='left') - 1
        z = (dj.r / dj.sig_cc).to_numpy(); Lj = (-dj.r).to_numpy()
        thr = pd.Series(Lj).shift(1).rolling(250).quantile(0.95).to_numpy(); x = np.where(np.isfinite(thr), (Lj > thr).astype(float), 0.0)
        ok = i >= 0; zz = np.full(len(ref_utc), np.nan); xx = np.zeros(len(ref_utc)); age = np.full(len(ref_utc), 99.0)
        zz[ok] = z[i[ok]]; xx[ok] = x[i[ok]]; age[ok] = (ref_dates[ok] - dj.index.values[i[ok]]).astype('timedelta64[D]').astype(float)
        Z[j] = (zz, xx, age)
    return Z


@njit(cache=True)
def tracker(s, a, eta, burn):
    T = len(s); q = np.full(T, np.nan); u = np.quantile(s[:burn], 1 - a)
    for t in range(burn, T):
        q[t] = u; u += eta * ((1.0 if s[t] > u else 0.0) - a)
    return q


@njit(cache=True)
def gct(s, X, a, gamma, eta0, eta1, r, burn, thetas, seed):
    """Gated conformal tracker with a fixed-mixture conformal Shiryaev-Roberts detector (tail calibrators)."""
    np.random.seed(seed)
    T = len(s); q = np.full(T, np.nan); st = np.zeros(T)
    q0 = np.quantile(s[:burn], 1 - a); q1 = np.quantile(s[:burn], 1 - a / 3.0)
    hist = np.empty(T); nh = 0; R = np.zeros(len(thetas)); hold = 0
    for t in range(T):
        x = X[t]; alarm = hold > 0
        if not alarm and np.isfinite(x):
            cnt = 0
            for i in range(nh):
                if hist[i] < x: cnt += 1
            u = (cnt + np.random.rand()) / (nh + 1.0); S = 0.0
            for j in range(len(thetas)):
                R[j] = (1.0 - thetas[j]) * (1.0 - u) ** (-thetas[j]) * (R[j] + 1.0); S += R[j] / len(thetas)
            hist[nh] = x; nh += 1
            if S >= gamma and t >= burn:
                hold = r; alarm = True
        if t >= burn:
            thr = q1 if alarm else q0
            q[t] = thr; st[t] = 1.0 if alarm else 0.0
            e = 1.0 if s[t] > thr else 0.0
            if alarm: q1 += eta1 * (e - a)
            else: q0 += eta0 * (e - a)
        if hold > 0:
            hold -= 1
            if hold == 0:
                nh = 0; R[:] = 0.0
    return q, st


@njit(cache=True)
def linear_buffer(s, F, a, eta_u, eta_w, W, burn):
    """Learned buffer q = u + <w, phi>: online linear quantile regression by projected subgradient steps (w in [0, W]^K)."""
    T, K = F.shape; q = np.full(T, np.nan); u = np.quantile(s[:burn], 1 - a); w = np.zeros(K)
    for t in range(burn, T):
        qq = u + np.dot(w, F[t]); q[t] = qq; e = 1.0 if s[t] > qq else 0.0
        u += eta_u * (e - a); w = np.minimum(np.maximum(w + eta_w * F[t] * (e - a), 0.0), W)
    return q


def qr_scorecaster(s, F, a, burn, window=1000, refit=21):
    """Rolling linear quantile regression forecast of the (1-a) quantile of s given [1, F], fit on past data only
    (window days, refit every `refit` days; first fit at t = burn)."""
    import statsmodels.api as sm
    T = len(s); out = np.full(T, np.nan); X = np.column_stack([np.ones(T), F]); beta = None
    for t in range(burn, T):
        if beta is None or (t - burn) % refit == 0:
            lo = max(0, t - window); y = s[lo:t]; Xt = X[lo:t]; ok = np.isfinite(y)
            try:
                beta = sm.QuantReg(y[ok], Xt[ok]).fit(q=1 - a, max_iter=2000).params
            except Exception:
                beta = np.r_[np.quantile(y[ok], 1 - a), np.zeros(X.shape[1] - 1)]
        out[t] = X[t] @ beta
    return out


@njit(cache=True)
def pid_p(s, shat, a, eta, burn):
    """Conformal PID with a scorecaster, proportional (quantile-tracking) term only: q = shat + u."""
    T = len(s); q = np.full(T, np.nan); u = 0.0
    for t in range(burn, T):
        qq = shat[t] + u; q[t] = qq; u += eta * ((1.0 if s[t] > qq else 0.0) - a)
    return q


@njit(cache=True)
def dtaci(s, shat, a, burn, window, gammas, I):
    """DtACI (Gibbs and Candes 2024, Alg. 1) on residuals r = s - shat: threshold shat + empirical (1 - alpha_t) quantile of
    the last `window` residuals; experts with steps `gammas`, sigma = 1/(2I), eta from their default formula."""
    T = len(s); K = len(gammas); q = np.full(T, np.nan); r = s - shat
    sig = 1.0 / (2 * I); eta = np.sqrt(3.0 / I) * np.sqrt((np.log(K * I) + 2) / ((1 - a) ** 2 * a ** 3))
    al = np.full(K, a); w = np.ones(K) / K
    for t in range(burn, T):
        lo = max(0, t - window); past = r[lo:t]; past = past[np.isfinite(past)]
        if len(past) < 250 or not np.isfinite(r[t]):
            continue
        at = np.sum(w * al); at = min(max(at, 1e-4), 0.9999)
        q[t] = shat[t] + np.quantile(past, 1 - at)
        # realised level beta: largest b with r_t <= Q_{1-b}(past)
        beta = np.mean(past >= r[t])
        for k in range(K):
            ak = min(max(al[k], 1e-4), 0.9999); err = 1.0 if r[t] > np.quantile(past, 1 - ak) else 0.0
            d = beta - al[k]; loss = a * d - min(0.0, d)
            w[k] = w[k] * np.exp(-eta * loss); al[k] = al[k] + gammas[k] * (a - err)
        w = w / np.sum(w); w = (1 - sig) * w + sig / K
    return q
