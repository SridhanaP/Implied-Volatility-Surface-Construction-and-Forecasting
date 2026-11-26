"""
Download long daily price histories needed for the forecasting study (Steps 8-9).

Run once (and again whenever you want to refresh the data):
    python src/download_history.py

Saves one CSV per series into data/history/
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "history"

# Yahoo symbol -> file name
SERIES = {
    "^GSPC": "sp500",      # S&P 500 index: used to compute realized volatility
    "SPY": "spy",          # S&P 500 fund: the underlying of our option chains
    "^VIX": "vix",         # 30-day implied volatility of the S&P 500
    "^VIX9D": "vix9d",     # 9-day implied volatility
    "^VIX3M": "vix3m",     # 3-month implied volatility
    "^IRX": "tbill_3m",    # 13-week T-bill yield in percent: our risk-free rate
}


def download_all(start="1990-01-01"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for symbol, name in SERIES.items():
        df = yf.download(symbol, start=start, auto_adjust=False, progress=False)

        if df.empty:
            print(f"{symbol:8s} -> no data returned, skipped")
            continue

        # Newer yfinance versions return two-level column names; flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        path = OUT_DIR / f"{name}.csv"
        df.to_csv(path)
        first, last = df.index[0].date(), df.index[-1].date()
        print(f"{symbol:8s} -> {len(df):6d} days ({first} to {last})  saved to {path.name}")


if __name__ == "__main__":
    download_all()
