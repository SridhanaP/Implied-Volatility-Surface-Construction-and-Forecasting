"""
Compute smile and term structure metrics, plot them, and build a daily history.

Run from the project folder with:
    python -m src.plot_metrics

Saves:
    data/processed/metrics_SPY_<date>.csv     one row per expiry (latest day)
    data/processed/metrics_history.csv        one row per collected day (30d and 91d metrics)
    plots/term_structure_SPY_<date>.png
"""
import matplotlib.pyplot as plt
import pandas as pd

from src.data_prep import DATA_DIR, build_iv_table, latest_chain_file
from src.metrics import compute_metrics, constant_maturity

PROCESSED = DATA_DIR / "processed"


def latest_vix():
    try:
        vix = pd.read_csv(DATA_DIR / "history" / "vix.csv", index_col=0)["Close"]
        vix = pd.to_numeric(vix, errors="coerce").dropna()
        return vix.index[-1], vix.iloc[-1] / 100
    except Exception:
        return None, None


def plot(m, title, out):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax[0, 0].plot(m["days"], m["atm_vol"] * 100, "o-", label="ATM implied vol")
    ax[0, 0].plot(m["days"], m["fwd_vol"] * 100, "s--", ms=4, alpha=0.7, label="forward vol between expiries")
    ax[0, 0].set(title="Term structure of ATM volatility", xlabel="days to expiry", ylabel="vol (%)")
    ax[0, 0].legend()

    bad = m[~m["calendar_ok"]]
    ax[0, 1].plot(m["T"], m["atm_total_var"], "o-")
    ax[0, 1].plot(bad["T"], bad["atm_total_var"], "rx", ms=12, mew=2, label="calendar arbitrage")
    ax[0, 1].set(title="ATM total variance  w = vol² × T  (must rise)", xlabel="T (years)", ylabel="w")
    if len(bad):
        ax[0, 1].legend()

    ax[1, 0].plot(m["days"], m["skew"], "o-", color="tab:red")
    ax[1, 0].axhline(0, color="grey", lw=0.8)
    ax[1, 0].set(title="ATM skew  (d vol / d k)", xlabel="days to expiry", ylabel="skew")

    ax[1, 1].plot(m["days"], m["rr25"] * 100, "o-", label="25Δ risk reversal")
    ax[1, 1].plot(m["days"], m["bf25"] * 100, "o-", label="25Δ butterfly")
    ax[1, 1].axhline(0, color="grey", lw=0.8)
    ax[1, 1].set(title="Delta-based smile measures", xlabel="days to expiry", ylabel="vol points (%)")
    ax[1, 1].legend()

    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def build_history():
    """30-day and 91-day metrics for every option file collected so far."""
    rows = []
    for path in sorted((DATA_DIR / "options").glob("SPY_*.csv")):
        try:
            m = compute_metrics(build_iv_table(path))
        except Exception as e:
            print(f"  skipped {path.name}: {e}")
            continue
        rows.append({"date": path.stem.split("_")[1], **constant_maturity(m, 30), **constant_maturity(m, 91)})
    return pd.DataFrame(rows)


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = latest_chain_file()
    print(f"Using {path.name}\n")
    m = compute_metrics(build_iv_table(path))
    m.to_csv(PROCESSED / f"metrics_{path.stem}.csv", index=False)

    show = m[["days", "atm_vol", "skew", "curvature", "rr25", "bf25", "fwd_vol", "calendar_ok"]]
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print(show.to_string(index=False))

    cm = constant_maturity(m, 30)
    vix_date, vix = latest_vix()
    print(f"\n30-day ATM vol (interpolated): {cm['atm_vol_30d']:.2%}")
    if vix is not None:
        print(f"VIX on {vix_date}:            {vix:.2%}   (VIX uses all strikes, so it is usually a bit higher)")
    print(f"30-day 25-delta risk reversal: {cm['rr25_30d']:.2%}")
    n_bad = (~m["calendar_ok"]).sum()
    print(f"Calendar arbitrage violations in ATM total variance: {n_bad}")

    history = build_history()
    history.to_csv(PROCESSED / "metrics_history.csv", index=False)
    print(f"\nHistory: {len(history)} day(s) saved to data/processed/metrics_history.csv")

    plots = DATA_DIR.parent / "plots"
    plots.mkdir(exist_ok=True)
    out = plots / f"term_structure_{path.stem}.png"
    plot(m, f"SPY smile and term structure metrics, {path.stem.split('_')[1]}", out)
    print(f"Chart saved to {out}\nClose the chart window to finish.")
    plt.show()


if __name__ == "__main__":
    main()
