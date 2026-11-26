"""
Turn a raw option chain CSV into a clean table of implied volatilities.

This is the "cleaning" half of Step 1, which needed Steps 2-3 to work:
  1. Clean the quotes: use mid prices, drop options with no bid, very wide
     spreads, or less than 7 days to expiry.
  2. For each expiry, measure the forward price F and discount factor DF
     from put-call parity:   Call - Put = DF * (F - K)
     A straight-line fit of (Call - Put) against strike K gives
     slope = -DF and intercept = DF * F. This captures interest rates AND
     dividends straight from market prices, without guessing either.
  3. Keep only out-of-the-money options (puts below F, calls above F):
     they are the most liquid and least affected by early exercise.
  4. Compute each option's implied volatility.

Note: SPY options are American-style (can be exercised early), while
Black-Scholes assumes European. For out-of-the-money options the
difference is very small, which is another reason we keep only those.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.iv import implied_vols

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def latest_chain_file(ticker="SPY"):
    files = sorted((DATA_DIR / "options").glob(f"{ticker}_*.csv"))
    if not files:
        raise FileNotFoundError("No option files in data/options. Run src/collect_options.py first.")
    return files[-1]


def risk_free_rate():
    """Latest 13-week T-bill yield as a decimal. Used only as a fallback."""
    try:
        closes = pd.read_csv(DATA_DIR / "history" / "tbill_3m.csv", index_col=0)["Close"]
        return float(pd.to_numeric(closes, errors="coerce").dropna().iloc[-1]) / 100
    except Exception:
        return 0.04


def clean_chain(raw, min_days=7, max_rel_spread=0.5):
    df = raw.copy()
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["days"] = (df["expiry"] - df["quote_date"]).dt.days
    df["T"] = df["days"] / 365
    df["is_call"] = df["type"] == "call"
    df["mid"] = (df["bid"] + df["ask"]) / 2

    has_quote = (df["bid"] > 0) & (df["ask"] > df["bid"])
    if has_quote.sum() < 50:
        print("WARNING: almost no live bid/ask quotes (was the market closed?).")
        print("         Using last traded prices instead; results will be noisier.")
        df["price"] = df["lastPrice"]
        good = df["price"] > 0
    else:
        df["price"] = df["mid"]
        good = has_quote & ((df["ask"] - df["bid"]) <= max_rel_spread * df["mid"])

    return df[good & (df["days"] >= min_days)].copy()


def estimate_forward(group, spot, r_fallback):
    """Forward price F and discount factor DF for one expiry, via put-call parity."""
    T = group["T"].iloc[0]
    DF_rate = np.exp(-r_fallback * T)

    calls = group[group["is_call"]].groupby("strike")["price"].mean()
    puts = group[~group["is_call"]].groupby("strike")["price"].mean()
    pairs = pd.concat([calls, puts], axis=1, keys=["C", "P"]).dropna()
    if pairs.empty:
        return spot / DF_rate, DF_rate

    # use the strikes closest to the spot price, where both quotes are reliable
    near = pairs.iloc[np.argsort(np.abs(pairs.index.values - spot))[:10]]

    if len(near) >= 3:
        slope, intercept = np.polyfit(near.index.values, (near["C"] - near["P"]).values, 1)
        DF = -slope
        if abs(DF - DF_rate) < 0.02:              # sanity check against T-bill rate
            F = intercept / DF
            if abs(F / spot - 1) < 0.1:
                return F, DF

    # fallback: DF from the T-bill rate, F from parity at the closest strike
    K0 = near.index[0]
    return K0 + (near.loc[K0, "C"] - near.loc[K0, "P"]) / DF_rate, DF_rate


def build_iv_table(path=None, min_days=7, k_range=(-0.5, 0.3)):
    """Full pipeline: raw CSV -> clean, out-of-the-money options with implied vols."""
    path = Path(path) if path else latest_chain_file()
    raw = pd.read_csv(path)
    spot = float(raw["spot"].iloc[0])

    df = clean_chain(raw, min_days=min_days)
    if df.empty:
        raise ValueError(f"No usable quotes left in {path.name} after cleaning.")

    r = risk_free_rate()
    parts = []
    for _, g in df.groupby("expiry"):
        F, DF = estimate_forward(g, spot, r)
        parts.append(g.assign(F=F, DF=DF))
    df = pd.concat(parts)

    df["k"] = np.log(df["strike"] / df["F"])   # log-moneyness
    otm = (df["is_call"] & (df["strike"] >= df["F"])) | (~df["is_call"] & (df["strike"] < df["F"]))
    df = df[otm & df["k"].between(*k_range)].copy()

    df["iv"] = implied_vols(df["price"].values, df["F"].values, df["strike"].values,
                            df["T"].values, df["DF"].values, df["is_call"].values)
    df = df.dropna(subset=["iv"])

    cols = ["quote_date", "expiry", "days", "T", "type", "strike", "k", "bid", "ask",
            "price", "F", "DF", "iv", "impliedVolatility", "volume", "openInterest"]
    return df[cols].sort_values(["expiry", "strike"]).reset_index(drop=True)
