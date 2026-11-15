"""Import manually downloaded Stooq files (data/manual/original/) for THA (SET) and PHL (PSEi).

Usage: .venv/bin/python 01b_manual_import.py <paper dir>
Keeps the originals untouched and writes data/manual/THA.csv and data/manual/PHL.csv restricted to
2000-01-01 .. 2025-12-31 (same span as the Yahoo downloads), columns Date, Open, High, Low, Close, Volume.
"""
import sys
from pathlib import Path

import pandas as pd

FILES = {'THA': 'set_d_stooq.csv', 'PHL': 'psei_d_stooq.csv'}

base = Path(sys.argv[1]).resolve()
for code, fn in FILES.items():
    d = pd.read_csv(base / 'data' / 'manual' / 'original' / fn, parse_dates=['Date'])
    d = d[(d.Date >= '2000-01-01') & (d.Date <= '2025-12-31')].sort_values('Date')
    d.to_csv(base / 'data' / 'manual' / f'{code}.csv', index=False)
    print(code, len(d), d.Date.min().date(), d.Date.max().date())
