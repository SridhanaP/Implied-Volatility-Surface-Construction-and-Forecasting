"""
Download today's option chain for SPY from Yahoo Finance and save it as a CSV.

Run once per trading day, from the project folder:
    python src/collect_options.py

Each run creates a file like data/options/SPY_2026-11-27.csv
"""
from datetime import date
from pathlib import Path
import time

import pandas as pd
import yfinance as yf

TICKER = "SPY"

# data/options/ inside the project folder (works no matter where you run it from)
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "options"


def get_spot(tk):
    """Latest closing price of the underlying."""
    hist = tk.history(period="5d")
    return float(hist["Close"].iloc[-1])


def collect(ticker=TICKER):
    tk = yf.Ticker(ticker)
    spot = get_spot(tk)
    today = date.today().isoformat()

    expiries = tk.options  # list of expiry dates, e.g. ['2026-11-27', '2026-12-04', ...]
    print(f"Found {len(expiries)} expiry dates for {ticker}. Downloading...")

    frames = []
    for expiry in expiries:
        try:
            chain = tk.option_chain(expiry)
        except Exception as e:  # one bad expiry shouldn't stop the whole download
            print(f"  skipped {expiry}: {e}")
            continue

        for df, kind in [(chain.calls, "call"), (chain.puts, "put")]:
            df = df.copy()
            df["type"] = kind
            df["expiry"] = expiry
            frames.append(df)

        time.sleep(0.3)  # be polite to Yahoo's servers

    data = pd.concat(frames, ignore_index=True)
    data["quote_date"] = today
    data["spot"] = spot

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{ticker}_{today}.csv"
    data.to_csv(path, index=False)

    print(f"Saved {len(data)} options across {data['expiry'].nunique()} expiries")
    print(f"Spot price: {spot:.2f}")
    print(f"File: {path}")
    return data


if __name__ == "__main__":
    collect()
