"""Download daily OHLC data for the study indices from Yahoo Finance (yfinance).

Usage: .venv/bin/python 01_download.py <paper dir>
Writes data/raw/<name>.csv (Date, Open, High, Low, Close, Volume) and data/raw/download_log.csv.
Raw files are never edited afterwards; cleaning happens in 02_clean.py.
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = {'IDX': '^JKSE', 'MYS': '^KLSE', 'THA': '^SET.BK', 'PHL': 'PSEI.PS', 'SGP': '^STI', 'USA': '^GSPC'}


def main():
    base = Path(sys.argv[1]).resolve()
    out = base / 'data' / 'raw'
    out.mkdir(parents=True, exist_ok=True)
    log = []
    for name, tic in TICKERS.items():
        df = yf.download(tic, start='2000-01-01', end='2026-01-01', interval='1d', auto_adjust=False, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(how='all')
        df.index.name = 'Date'
        df.to_csv(out / f'{name}.csv')
        log.append({'name': name, 'ticker': tic, 'rows': len(df), 'first': df.index.min().date() if len(df) else None,
                    'last': df.index.max().date() if len(df) else None, 'downloaded': date.today().isoformat()})
        print(log[-1])
    pd.DataFrame(log).to_csv(out / 'download_log.csv', index=False)


if __name__ == '__main__':
    main()
