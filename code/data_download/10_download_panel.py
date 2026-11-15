"""Fresh confirmatory panel: indices NOT previously examined in this project. Downloads OHLC only; no performance is computed.
Inclusion rule fixed here, before any forecast exists: history from 2001-12-31 or earlier, data through Dec 2025, no gap over 30 calendar days in
2005-2025, zero-return share under 2%, at least 225 observations per calendar year on average in 2005-2025."""
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf
OUT = Path('/home/nafisnaufal1426/aciids/astar/data/raw')
CAND = {
 # Asia-Pacific
 'JPN': ('^N225', 'APAC'), 'HKG': ('^HSI', 'APAC'), 'KOR': ('^KS11', 'APAC'), 'TWN': ('^TWII', 'APAC'), 'AUS': ('^AXJO', 'APAC'),
 'IND': ('^BSESN', 'APAC'), 'CHN': ('000001.SS', 'APAC'), 'NZL': ('^NZ50', 'APAC'),
 # Europe
 'GBR': ('^FTSE', 'EUR'), 'DEU': ('^GDAXI', 'EUR'), 'FRA': ('^FCHI', 'EUR'), 'ESP': ('^IBEX', 'EUR'), 'CHE': ('^SSMI', 'EUR'), 'NLD': ('^AEX', 'EUR'),
 'BEL': ('^BFX', 'EUR'), 'SWE': ('^OMX', 'EUR'), 'AUT': ('^ATX', 'EUR'), 'ITA': ('FTSEMIB.MI', 'EUR'),
 # Americas
 'CAN': ('^GSPTSE', 'AMER'), 'BRA': ('^BVSP', 'AMER'), 'MEX': ('^MXX', 'AMER'), 'ARG': ('^MERV', 'AMER'),
 # Middle East / Africa
 'ISR': ('^TA125.TA', 'MEA'), 'ZAF': ('^J203.JO', 'MEA'), 'EGY': ('^CASE30', 'MEA'), 'TUR': ('XU100.IS', 'MEA'),
}
rows = []
for name, (tic, reg) in CAND.items():
    try:
        df = yf.download(tic, start='2000-01-01', end='2026-01-01', interval='1d', auto_adjust=False, progress=False, threads=False)
    except Exception as ex:
        rows.append(dict(name=name, ticker=tic, region=reg, status=f'error {type(ex).__name__}')); continue
    if df is None or len(df) == 0:
        rows.append(dict(name=name, ticker=tic, region=reg, status='empty')); continue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(how='all'); df.index.name = 'Date'
    df.to_csv(OUT / f'{name}.csv')
    c = df['Close'].dropna(); w = c.loc['2005':'2025']; r = np.log(c).diff().dropna()
    gaps = w.index.to_series().diff().dt.days.max() if len(w) else np.nan
    ok = (len(c) > 0 and c.index.min() <= pd.Timestamp('2001-12-31') and c.index.max() >= pd.Timestamp('2025-12-01') and gaps <= 30
          and (r.loc['2005':'2025'] == 0).mean() < 0.02 and len(w) / 21 >= 225)
    rows.append(dict(name=name, ticker=tic, region=reg, status='ok' if ok else 'fail', first=c.index.min().date() if len(c) else None, last=c.index.max().date() if len(c) else None,
                     n=len(c), max_gap=gaps, zero_share=round((r.loc['2005':'2025'] == 0).mean(), 4), per_year=round(len(w) / 21, 1), downloaded=date.today().isoformat()))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT.parent / 'panel_screen.csv', index=False)
print(pd.DataFrame(rows)[['name', 'ticker', 'region', 'status', 'first', 'last', 'n', 'max_gap', 'zero_share', 'per_year']].to_string())
