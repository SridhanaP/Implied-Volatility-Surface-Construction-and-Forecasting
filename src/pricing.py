"""
Black-Scholes option pricing (Step 2).

Two equivalent ways to write the formula:

1. Spot form (the textbook version), using the stock price S,
   interest rate r and dividend yield q:              bs_price(S, K, T, r, sigma, q)

2. Forward form (Black-76), using the forward price F and the
   discount factor DF:                                black_price(F, K, T, sigma, DF)

They give identical prices, because F = S * exp((r - q) * T) and DF = exp(-r * T).
The forward form is cleaner for building volatility surfaces, since we will
measure F and DF directly from market prices in Step 1's cleaning stage.

All functions accept single numbers or numpy arrays (e.g. many strikes at once).
Units: T in years, r / q / sigma as decimals (5% = 0.05, 20% vol = 0.20).
"""
import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------- helpers

def _check(T, sigma):
    if np.any(np.asarray(T) <= 0):
        raise ValueError("Time to expiry T must be positive.")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("Volatility sigma must be positive.")


def _out(x):
    """Return a plain float for single inputs, an array for array inputs."""
    return x.item() if np.ndim(x) == 0 else x


def _d1_d2(F, K, T, sigma):
    std = sigma * np.sqrt(T)              # total standard deviation to expiry
    d1 = (np.log(F / K) + 0.5 * std**2) / std
    d2 = d1 - std
    return d1, d2


def forward(S, T, r, q=0.0):
    """Forward price of the underlying: S * exp((r - q) T)."""
    return S * np.exp((r - q) * T)


def discount(T, r):
    """Discount factor: value today of 1 paid at time T."""
    return np.exp(-r * T)


# ---------------------------------------------------------- forward form

def black_price(F, K, T, sigma, DF=1.0, is_call=True):
    """Black-76 price of a European option on a forward."""
    _check(T, sigma)
    F, K, T, sigma, DF = map(np.asarray, (F, K, T, sigma, DF))
    d1, d2 = _d1_d2(F, K, T, sigma)
    call = DF * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = DF * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return _out(np.where(is_call, call, put))


def black_vega(F, K, T, sigma, DF=1.0):
    """dPrice/dSigma in the forward form. Same for calls and puts.
    Used by the implied volatility solver in Step 3."""
    _check(T, sigma)
    F, K, T, sigma, DF = map(np.asarray, (F, K, T, sigma, DF))
    d1, _ = _d1_d2(F, K, T, sigma)
    return _out(DF * F * norm.pdf(d1) * np.sqrt(T))


# ------------------------------------------------------------- spot form

def bs_price(S, K, T, r, sigma, q=0.0, is_call=True):
    """Black-Scholes-Merton price of a European option."""
    return black_price(forward(S, T, r, q), K, T, sigma, discount(T, r), is_call)


def delta(S, K, T, r, sigma, q=0.0, is_call=True):
    """dPrice/dS: how much the option price moves per 1 unit move in S."""
    _check(T, sigma)
    d1, _ = _d1_d2(forward(S, T, r, q), K, T, sigma)
    carry = np.exp(-q * T)
    return _out(np.where(is_call, carry * norm.cdf(d1), carry * (norm.cdf(d1) - 1.0)))


def gamma(S, K, T, r, sigma, q=0.0):
    """d2Price/dS2: how fast delta changes. Same for calls and puts."""
    _check(T, sigma)
    d1, _ = _d1_d2(forward(S, T, r, q), K, T, sigma)
    return _out(np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def vega(S, K, T, r, sigma, q=0.0):
    """dPrice/dSigma, per 1.00 change in vol (divide by 100 for 'per 1 vol point')."""
    return black_vega(forward(S, T, r, q), K, T, sigma, discount(T, r))
