"""
Tests for src/pricing.py.

The main idea: price the same option with our own code and with QuantLib
(a professional library), and check that the answers agree.

Run from the project folder with:   pytest -v
"""
import numpy as np
import pytest
import QuantLib as ql

from src.pricing import bs_price, black_price, black_vega, delta, gamma, vega, forward, discount

# Test cases: (S, K, days to expiry, r, q, sigma)
# Days are used so that T = days / 365 matches QuantLib's calendar exactly.
CASES = [
    (100.0, 100.0, 365, 0.05, 0.02, 0.20),    # at the money, 1 year
    (100.0, 80.0, 182, 0.03, 0.00, 0.35),     # in/out of the money, 6 months
    (100.0, 130.0, 730, 0.04, 0.01, 0.25),    # far strike, 2 years
    (4500.0, 4000.0, 30, 0.05, 0.015, 0.15),  # index-like, 1 month
]


def quantlib_option(S, K, days, r, q, sigma, is_call):
    """Build the same option in QuantLib and return its price and greeks."""
    today = ql.Date(2, 1, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()

    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    r_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc))
    q_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc))
    vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sigma, dc))
    process = ql.BlackScholesMertonProcess(spot, q_curve, r_curve, vol)

    kind = ql.Option.Call if is_call else ql.Option.Put
    option = ql.VanillaOption(ql.PlainVanillaPayoff(kind, K), ql.EuropeanExercise(today + days))
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option.NPV(), option.delta(), option.gamma(), option.vega()


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("S, K, days, r, q, sigma", CASES)
def test_matches_quantlib(S, K, days, r, q, sigma, is_call):
    T = days / 365
    ql_price, ql_delta, ql_gamma, ql_vega = quantlib_option(S, K, days, r, q, sigma, is_call)

    assert bs_price(S, K, T, r, sigma, q, is_call) == pytest.approx(ql_price, rel=1e-7)
    assert delta(S, K, T, r, sigma, q, is_call) == pytest.approx(ql_delta, rel=1e-7)
    assert gamma(S, K, T, r, sigma, q) == pytest.approx(ql_gamma, rel=1e-7)
    assert vega(S, K, T, r, sigma, q) == pytest.approx(ql_vega, rel=1e-7)


@pytest.mark.parametrize("S, K, days, r, q, sigma", CASES)
def test_black_formula_matches_quantlib(S, K, days, r, q, sigma):
    T = days / 365
    F, DF = forward(S, T, r, q), discount(T, r)
    expected = ql.blackFormula(ql.Option.Call, K, F, sigma * np.sqrt(T), DF)
    assert black_price(F, K, T, sigma, DF, is_call=True) == pytest.approx(expected, rel=1e-7)


def test_textbook_value():
    # Classic example: S=K=100, T=1, r=5%, sigma=20%, no dividends -> call = 10.4506
    assert bs_price(100, 100, 1.0, 0.05, 0.20) == pytest.approx(10.4506, abs=1e-4)
    assert bs_price(100, 100, 1.0, 0.05, 0.20, is_call=False) == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("S, K, days, r, q, sigma", CASES)
def test_put_call_parity(S, K, days, r, q, sigma):
    # Call - Put = DF * (F - K), must hold for any volatility
    T = days / 365
    call = bs_price(S, K, T, r, sigma, q, True)
    put = bs_price(S, K, T, r, sigma, q, False)
    assert call - put == pytest.approx(discount(T, r) * (forward(S, T, r, q) - K), abs=1e-10)


def test_price_stays_within_no_arbitrage_bounds():
    F, DF, T = 100.0, 0.97, 0.5
    strikes = np.linspace(50, 150, 21)
    calls = black_price(F, strikes, T, 0.3, DF)
    assert np.all(calls >= DF * np.maximum(F - strikes, 0) - 1e-12)
    assert np.all(calls <= DF * F)


def test_vega_matches_finite_difference():
    F, K, T, sigma, DF, h = 100.0, 110.0, 0.75, 0.25, 0.98, 1e-5
    bumped = (black_price(F, K, T, sigma + h, DF) - black_price(F, K, T, sigma - h, DF)) / (2 * h)
    assert black_vega(F, K, T, sigma, DF) == pytest.approx(bumped, rel=1e-6)


def test_works_with_arrays():
    strikes = np.array([90.0, 100.0, 110.0])
    prices = bs_price(100.0, strikes, 1.0, 0.05, 0.2)
    assert prices.shape == (3,)
    assert np.all(np.diff(prices) < 0)  # call prices fall as strike rises


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        bs_price(100, 100, 0.0, 0.05, 0.2)    # zero time to expiry
    with pytest.raises(ValueError):
        bs_price(100, 100, 1.0, 0.05, -0.2)   # negative volatility
