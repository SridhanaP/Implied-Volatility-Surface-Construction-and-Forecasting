"""
Implied volatility (Step 3).

Black-Scholes turns a volatility into a price. Implied volatility goes the
other way: given the market price, find the volatility that makes the
Black-Scholes price equal to it.

There is no formula for this, so we solve it numerically:
  1. Check the price is possible at all (within no-arbitrage bounds).
     If not, no volatility can produce it, so we return NaN.
  2. Try Newton's method: fast, uses vega (the slope of price vs volatility).
  3. If Newton struggles (e.g. deep out-of-the-money options where vega is
     almost zero), fall back to Brent's method: slower, but guaranteed to
     converge because price always increases with volatility.
"""
import numpy as np
from scipy.optimize import brentq

from src.pricing import black_price, black_vega

SIGMA_LOW, SIGMA_HIGH = 1e-6, 5.0   # search range: 0.0001% to 500% volatility


def price_bounds(F, K, DF, is_call):
    """No-arbitrage range for an option price (forward form)."""
    if is_call:
        return DF * max(F - K, 0.0), DF * F
    return DF * max(K - F, 0.0), DF * K


def implied_vol(price, F, K, T, DF=1.0, is_call=True, tol=1e-10, max_newton=50):
    """Implied volatility of one European option. Returns NaN if none exists."""
    if not np.isfinite(price) or T <= 0:
        return np.nan

    lower, upper = price_bounds(F, K, DF, is_call)
    if price <= lower or price >= upper:
        return np.nan

    def error(sigma):
        return black_price(F, K, T, sigma, DF, is_call) - price

    # --- Newton's method, starting at the point where it converges best
    sigma = max(np.sqrt(2 * abs(np.log(F / K)) / T), 0.2)
    for _ in range(max_newton):
        diff = error(sigma)
        if abs(diff) < tol * max(price, 1.0):
            return float(sigma)
        v = black_vega(F, K, T, sigma, DF)
        if v < 1e-12:
            break                       # slope too flat: Newton would jump wildly
        new_sigma = sigma - diff / v
        if not SIGMA_LOW < new_sigma < SIGMA_HIGH:
            break                       # stepped outside the search range
        sigma = new_sigma

    # --- Brent's method fallback (guaranteed, since price rises with sigma)
    if error(SIGMA_HIGH) < 0:
        return np.nan                   # would need more than 500% vol
    return float(brentq(error, SIGMA_LOW, SIGMA_HIGH, xtol=1e-12, maxiter=500))


def implied_vols(prices, F, K, T, DF=1.0, is_call=True):
    """Implied volatility for many options at once (arrays of equal length)."""
    arrays = np.broadcast_arrays(*map(np.asarray, (prices, F, K, T, DF, is_call)))
    flat = zip(*(a.ravel() for a in arrays))
    out = np.array([implied_vol(p, f, k, t, d, bool(c)) for p, f, k, t, d, c in flat])
    return out.reshape(arrays[0].shape)
