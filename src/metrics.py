"""
Smile, skew and term structure metrics (Step 4).

Takes the implied vol table from src/data_prep.py and summarises each expiry
with the numbers traders use to describe a volatility smile:

  atm_vol     at-the-money implied vol (at k = 0, i.e. strike = forward)
  skew        slope of the smile at the money (d vol / d k). Negative for
              equity indices: downside puts are more expensive than calls.
  curvature   how strongly the smile bends upwards (the "smile" part)
  put25       implied vol of the 25-delta put  (a moderately out-of-the-money put)
  call25      implied vol of the 25-delta call
  rr25        risk reversal  = call25 - put25. Measures skew in delta terms.
  bf25        butterfly      = (call25 + put25) / 2 - atm_vol. Measures curvature.

And across expiries (the term structure):

  atm_total_var  total implied variance w = atm_vol^2 * T. Must never decrease
                 as T grows, otherwise there is calendar arbitrage.
  fwd_vol        the volatility the market implies BETWEEN two expiries:
                 sqrt((w2 - w1) / (T2 - T1)).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def forward_delta(k, T, iv, is_call):
    """Black forward delta of an option, from its log-moneyness and own implied vol."""
    d1 = (-k + 0.5 * iv**2 * T) / (iv * np.sqrt(T))
    return np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)


def vol_at_delta(g, target):
    """Implied vol at a given delta (e.g. +0.25 call, -0.25 put) by interpolation."""
    is_call = target > 0
    side = g[g["type"] == ("call" if is_call else "put")]
    if len(side) < 2:
        return np.nan
    d = forward_delta(side["k"].values, side["T"].values, side["iv"].values, is_call)
    order = np.argsort(d)
    d, v = d[order], side["iv"].values[order]
    if not d[0] <= target <= d[-1]:
        return np.nan
    return float(np.interp(target, d, v))


def slice_metrics(g):
    """All smile metrics for ONE expiry."""
    T = float(g["T"].iloc[0])

    # quadratic fit near the money: iv = atm + skew * k + curvature * k^2
    # the window widens for longer expiries, whose smiles are spread wider
    band = max(0.05, 0.15 * np.sqrt(T))
    near = g[g["k"].abs() <= band]
    if len(near) >= 5:
        curvature, skew, atm = np.polyfit(near["k"], near["iv"], 2)
    else:
        curvature = skew = atm = np.nan

    put25, call25 = vol_at_delta(g, -0.25), vol_at_delta(g, 0.25)
    return {
        "expiry": g["expiry"].iloc[0],
        "days": int(g["days"].iloc[0]),
        "T": T,
        "F": float(g["F"].iloc[0]),
        "n_options": len(g),
        "atm_vol": atm,
        "skew": skew,
        "curvature": curvature,
        "put25": put25,
        "call25": call25,
        "rr25": call25 - put25,
        "bf25": (call25 + put25) / 2 - atm,
        "atm_total_var": atm**2 * T,
    }


def compute_metrics(table):
    """Metrics for every expiry, plus term structure columns."""
    m = pd.DataFrame([slice_metrics(g) for _, g in table.groupby("expiry")])
    m = m.sort_values("T").reset_index(drop=True)

    dw = m["atm_total_var"].diff()
    dT = m["T"].diff()
    m["calendar_ok"] = (dw >= 0) | dw.isna()           # total variance must not fall
    fwd_var = (dw / dT).where(dw >= 0)
    fwd_var.iloc[0] = m["atm_total_var"].iloc[0] / m["T"].iloc[0]
    m["fwd_vol"] = np.sqrt(fwd_var)
    return m


def constant_maturity(m, days=30):
    """Metrics at a fixed maturity (e.g. 30 days), interpolated between expiries.

    Expiries move closer every day, so to build a time series we need a fixed
    horizon. ATM vol is interpolated in total variance (the arbitrage-consistent
    way); the other metrics linearly in T. Returns NaN if `days` is not bracketed.
    """
    T = days / 365
    m = m.dropna(subset=["atm_vol"])
    if m.empty or not m["T"].iloc[0] <= T <= m["T"].iloc[-1]:
        return {f"{c}_{days}d": np.nan for c in ("atm_vol", "skew", "rr25", "bf25")}

    w = np.interp(T, m["T"], m["atm_total_var"])
    out = {f"atm_vol_{days}d": np.sqrt(w / T)}
    for c in ("skew", "rr25", "bf25"):
        valid = m.dropna(subset=[c])
        out[f"{c}_{days}d"] = float(np.interp(T, valid["T"], valid[c])) if len(valid) >= 2 else np.nan
    return out
