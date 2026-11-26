"""
Tests for src/iv.py and the forward estimation in src/data_prep.py.

Run from the project folder with:   pytest -v
"""
import numpy as np
import pandas as pd
import pytest
import QuantLib as ql

from src.data_prep import estimate_forward
from src.iv import implied_vol, implied_vols
from src.pricing import black_price

# (F, K, T, DF, sigma): at the money, out of the money, short and long dated
CASES = [
    (100.0, 100.0, 1.0, 0.95, 0.20),
    (100.0, 80.0, 0.5, 0.98, 0.35),
    (100.0, 125.0, 2.0, 0.92, 0.25),
    (4500.0, 4000.0, 0.08, 0.996, 0.15),
    (100.0, 160.0, 0.25, 0.99, 0.40),    # deep out of the money: tiny vega
    (100.0, 100.0, 0.02, 1.0, 0.80),     # one week, high vol
]


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("F, K, T, DF, sigma", CASES)
def test_round_trip(F, K, T, DF, sigma, is_call):
    # price with a known vol, then recover that vol from the price
    price = black_price(F, K, T, sigma, DF, is_call)
    assert implied_vol(price, F, K, T, DF, is_call) == pytest.approx(sigma, abs=1e-8)


@pytest.mark.parametrize("F, K, T, DF, sigma", CASES[:4])
def test_matches_quantlib(F, K, T, DF, sigma):
    price = black_price(F, K, T, sigma, DF, is_call=True)
    std = ql.blackFormulaImpliedStdDev(ql.Option.Call, K, F, price, DF, 0.0,
                                       0.2 * np.sqrt(T), 1e-12, 1000)
    assert implied_vol(price, F, K, T, DF, True) == pytest.approx(std / np.sqrt(T), abs=1e-6)


def test_impossible_prices_return_nan():
    F, K, T, DF = 100.0, 90.0, 1.0, 0.97
    assert np.isnan(implied_vol(DF * 10 - 0.01, F, K, T, DF, True))  # below intrinsic value
    assert np.isnan(implied_vol(DF * F + 1.0, F, K, T, DF, True))    # above the forward
    assert np.isnan(implied_vol(-1.0, F, K, T, DF, True))            # negative price
    assert np.isnan(implied_vol(np.nan, F, K, T, DF, True))          # missing price


def test_vectorized_smile():
    F, T, DF = 100.0, 0.5, 0.98
    strikes = np.linspace(70, 130, 13)
    true_vols = 0.2 + 0.3 * np.log(strikes / F) ** 2 - 0.1 * np.log(strikes / F)  # a smile
    is_call = strikes >= F
    prices = black_price(F, strikes, T, true_vols, DF, is_call)
    assert implied_vols(prices, F, strikes, T, DF, is_call) == pytest.approx(true_vols, abs=1e-8)


def test_forward_estimate_from_put_call_parity():
    # build a fake chain with a known forward and discount factor
    F_true, DF_true, T = 452.30, 0.9876, 0.25
    strikes = np.arange(420.0, 486.0, 5.0)
    rows = []
    for K in strikes:
        for is_call in (True, False):
            rows.append({"strike": K, "is_call": is_call, "T": T,
                         "price": black_price(F_true, K, T, 0.18, DF_true, is_call)})
    F, DF = estimate_forward(pd.DataFrame(rows), spot=450.0, r_fallback=0.05)
    assert F == pytest.approx(F_true, abs=1e-6)
    assert DF == pytest.approx(DF_true, abs=1e-8)
