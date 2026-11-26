"""
Tests for src/metrics.py, using fake smiles where the right answers are known.

Run from the project folder with:   pytest -v
"""
import numpy as np
import pandas as pd
import pytest

from src.metrics import compute_metrics, constant_maturity, forward_delta, vol_at_delta


def make_table(days_list, atm=0.20, skew=-0.30, curv=0.50):
    """Fake implied vol table with smile  iv = atm + skew*k + curv*k^2  for each expiry."""
    rows = []
    for days in days_list:
        T = days / 365
        for k in np.linspace(-0.4, 0.3, 141):
            rows.append({"expiry": pd.Timestamp("2027-01-01") + pd.Timedelta(days=days),
                         "days": days, "T": T, "F": 100.0, "k": k,
                         "type": "call" if k >= 0 else "put",
                         "iv": atm + skew * k + curv * k**2})
    return pd.DataFrame(rows)


def test_recovers_known_smile_shape():
    m = compute_metrics(make_table([30, 90, 180]))
    assert m["atm_vol"].values == pytest.approx(0.20, abs=1e-8)
    assert m["skew"].values == pytest.approx(-0.30, abs=1e-8)
    assert m["curvature"].values == pytest.approx(0.50, abs=1e-6)


def test_flat_smile_has_zero_risk_reversal_and_butterfly():
    m = compute_metrics(make_table([30, 90], skew=0.0, curv=0.0))
    assert m["rr25"].values == pytest.approx(0.0, abs=1e-10)
    assert m["bf25"].values == pytest.approx(0.0, abs=1e-10)


def test_equity_style_skew_gives_negative_risk_reversal():
    m = compute_metrics(make_table([30, 90]))
    assert np.all(m["rr25"] < 0)     # puts dearer than calls
    assert np.all(m["bf25"] > 0)     # wings dearer than the middle


def test_forward_delta_basics():
    # an at-the-money call has delta a little above 0.5; put = call - 1
    call = forward_delta(0.0, 1.0, 0.2, True)
    put = forward_delta(0.0, 1.0, 0.2, False)
    assert 0.5 < call < 0.55
    assert put == pytest.approx(call - 1)


def test_vol_at_delta_on_flat_smile():
    g = make_table([60], skew=0.0, curv=0.0)
    assert vol_at_delta(g, 0.25) == pytest.approx(0.20)
    assert vol_at_delta(g, -0.25) == pytest.approx(0.20)


def test_flat_term_structure_has_no_calendar_arbitrage():
    m = compute_metrics(make_table([10, 30, 90, 365]))
    assert m["calendar_ok"].all()
    assert m["fwd_vol"].values == pytest.approx(0.20, abs=1e-8)   # flat vol -> forward vol equal


def test_falling_total_variance_is_flagged():
    t1 = make_table([30], atm=0.40)
    t2 = make_table([60], atm=0.20)  # 0.2^2*60 < 0.4^2*30: total variance falls
    m = compute_metrics(pd.concat([t1, t2]))
    assert not m["calendar_ok"].iloc[1]


def test_constant_maturity_interpolation():
    m = compute_metrics(make_table([20, 40]))
    cm = constant_maturity(m, 30)
    assert cm["atm_vol_30d"] == pytest.approx(0.20, abs=1e-8)
    assert np.isnan(constant_maturity(m, 100)["atm_vol_100d"])   # 100 days is outside the data
