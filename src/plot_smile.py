"""
Compute implied volatilities for the latest SPY option chain and plot the smiles.

Run from the project folder with:
    python -m src.plot_smile

Saves:
    data/processed/iv_SPY_<date>.csv   (the full implied vol table)
    plots/smile_SPY_<date>.png         (the chart)
"""
import matplotlib.pyplot as plt
import numpy as np

from src.data_prep import DATA_DIR, build_iv_table, latest_chain_file


def main():
    path = latest_chain_file()
    print(f"Using {path.name}")
    table = build_iv_table(path)

    # save the table
    processed = DATA_DIR / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    table.to_csv(processed / f"iv_{path.stem}.csv", index=False)

    # summary per expiry: forward, discount factor and at-the-money vol
    print(f"\n{'expiry':>12} {'days':>5} {'forward':>9} {'DF':>8} {'options':>8} {'ATM vol':>8}")
    for expiry, g in table.groupby("expiry"):
        atm = g.loc[g["k"].abs().idxmin(), "iv"]
        print(f"{expiry.date()!s:>12} {g['days'].iloc[0]:5d} {g['F'].iloc[0]:9.2f} "
              f"{g['DF'].iloc[0]:8.5f} {len(g):8d} {atm:8.1%}")

    # plot up to 6 expiries, spread from shortest to longest
    expiries = sorted(table["expiry"].unique())
    picks = [expiries[i] for i in np.unique(np.linspace(0, len(expiries) - 1, min(6, len(expiries))).round().astype(int))]

    fig, ax = plt.subplots(figsize=(10, 6))
    for expiry in picks:
        g = table[table["expiry"] == expiry]
        ax.plot(g["k"], g["iv"] * 100, "o-", ms=3, lw=1, label=f"{g['days'].iloc[0]} days")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("log-moneyness  k = ln(K / F)     (left: puts, right: calls)")
    ax.set_ylabel("implied volatility (%)")
    ax.set_title(f"SPY implied volatility smiles, {table['quote_date'].iloc[0].date()}")
    ax.legend(title="time to expiry")
    ax.grid(alpha=0.3)

    plots = DATA_DIR.parent / "plots"
    plots.mkdir(exist_ok=True)
    out = plots / f"smile_{path.stem}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {len(table)} implied vols and the chart to {out}")
    plt.show()


if __name__ == "__main__":
    main()
