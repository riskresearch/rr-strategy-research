import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf
from pathlib import Path

# =========================================================
# Paths
# =========================================================
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = Path(__file__).parent.parent / "outputs" / "ch01"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHILLER_FILE = DATA_DIR / "shiller_ie_data.xls"

# =========================================================
# Style
# =========================================================
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

COLORS = {
    "yield":     "#2166ac",
    "growth":    "#4dac26",
    "inflation": "#d6604d",
    "multiple":  "#756bb1",
    "total":     "#1a1a1a",
    "scatter":   "#2166ac",
    "fit":       "#d6604d",
    "equity":    "#2166ac",
    "real_rate": "#d6604d",
    "neutral":   "#888888",
}


# =========================================================
# Load and clean Shiller data
# =========================================================
def load_shiller() -> pd.DataFrame:
    """
    Shiller date format is YYYY.MM  e.g. 1871.01 = Jan 1871
    The two digits after the decimal are the month number directly,
    NOT a fractional year.  Parse accordingly.
    """
    df = pd.read_excel(SHILLER_FILE, sheet_name="Data", header=7)
    df = df.iloc[:, :7].copy()
    df.columns = ["date_raw", "price", "dividend", "earnings",
                  "cpi", "col5", "cape"]

    # Keep only numeric rows
    df = df[pd.to_numeric(df["date_raw"], errors="coerce").notna()].copy()
    df["date_raw"] = df["date_raw"].astype(float).round(2)

    # Extract year and month correctly
    # e.g. 1871.01 -> year=1871, month=1
    #      1871.10 -> year=1871, month=10
    year_int  = df["date_raw"].astype(int)
    # Multiply fractional part by 100 to get month as integer
    month_int = ((df["date_raw"] - year_int) * 100).round(0).astype(int)
    month_int = month_int.clip(1, 12)

    df["date"] = pd.to_datetime(
        dict(year=year_int, month=month_int, day=1)
    )
    df = df.set_index("date").sort_index()

    for col in ["price", "dividend", "earnings", "cpi", "cape"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["price", "dividend", "cpi"])

    # Drop any remaining duplicates keeping first
    df = df[~df.index.duplicated(keep="first")]

    return df


# =========================================================
# Chart 1 — CAPE vs subsequent 10-year real return scatter
# =========================================================
def chart_cape_vs_returns(df: pd.DataFrame) -> None:
    df = df.copy()
    last_cpi = float(df["cpi"].iloc[-1])
    df["real_price"]    = df["price"]    / df["cpi"] * last_cpi
    df["real_dividend"] = df["dividend"] / df["cpi"] * last_cpi

    horizon = 120  # 10 years in months
    dates   = df.index.tolist()
    records = []

    for i in range(len(dates) - horizon):
        start = dates[i]
        end   = dates[i + horizon]

        cape_start = float(df.loc[start, "cape"])
        if pd.isna(cape_start) or cape_start <= 0:
            continue

        p0 = float(df.loc[start, "real_price"])
        p1 = float(df.loc[end,   "real_price"])
        if pd.isna(p0) or pd.isna(p1) or p0 <= 0:
            continue

        div_yield = float(df.loc[start, "dividend"]) / float(df.loc[start, "price"])
        price_ret = (p1 / p0) ** (1 / 10) - 1
        total_ret = price_ret + div_yield

        records.append({
            "date":  start,
            "cape":  cape_start,
            "ret10": total_ret * 100,
        })

    scatter = pd.DataFrame(records).dropna()

    x      = scatter["cape"].values
    y      = scatter["ret10"].values
    coeffs = np.polyfit(x, y, 1)
    x_fit  = np.linspace(x.min(), x.max(), 200)
    y_fit  = np.polyval(coeffs, x_fit)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        scatter["cape"], scatter["ret10"],
        s=8, alpha=0.5, color=COLORS["scatter"],
        label="Monthly observations",
    )
    ax.plot(
        x_fit, y_fit,
        color=COLORS["fit"], lw=1.8,
        label=f"OLS fit (slope = {coeffs[0]:.2f})",
    )

    for label, date_str, offset in [
        ("Jan 2000\n(CAPE≈44)", "2000-01-01", ( 2,  3)),
        ("Jan 1982\n(CAPE≈8)",  "1982-01-01", ( 2, -4)),
        ("Jan 2009\n(CAPE≈15)", "2009-01-01", ( 2,  3)),
    ]:
        row = scatter[scatter["date"] == pd.Timestamp(date_str)]
        if not row.empty:
            ax.annotate(
                label,
                xy=(float(row["cape"].iloc[0]), float(row["ret10"].iloc[0])),
                xytext=(
                    float(row["cape"].iloc[0]) + offset[0],
                    float(row["ret10"].iloc[0]) + offset[1],
                ),
                fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
                color="#333333",
            )

    ax.axhline(0, color="#999999", lw=0.8, linestyle="--")
    ax.set_xlabel("Shiller CAPE at start of period", labelpad=8)
    ax.set_ylabel("Subsequent 10-year annualized real total return (%)", labelpad=8)
    ax.set_title(
        "Starting valuation and long-horizon equity returns\n"
        "S&P 500, monthly observations 1881–2015",
        pad=12,
    )
    ax.legend(fontsize=8, framealpha=0.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_cape_vs_returns.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_cape_vs_returns.png")


# =========================================================
# Chart 2 — Return decomposition across regimes
# =========================================================
def chart_return_decomposition(df: pd.DataFrame) -> None:
    regimes = [
        ("1926–1965", "1926-01-01", "1965-12-01"),
        ("1966–1981", "1966-01-01", "1981-12-01"),
        ("1982–1999", "1982-01-01", "1999-12-01"),
        ("2000–2021", "2000-01-01", "2021-12-01"),
    ]

    rows = []
    for name, start, end in regimes:
        sub = df.loc[start:end].dropna(
            subset=["price", "dividend", "earnings", "cpi"]
        )
        if len(sub) < 24:
            continue

        n_years = len(sub) / 12
        p0      = float(sub["price"].iloc[0])
        p1      = float(sub["price"].iloc[-1])
        d0      = float(sub["dividend"].iloc[0])
        e0      = float(sub["earnings"].iloc[0])
        e1      = float(sub["earnings"].iloc[-1])
        cpi0    = float(sub["cpi"].iloc[0])
        cpi1    = float(sub["cpi"].iloc[-1])

        div_yield = (d0 / p0) * 100
        inflation = ((cpi1 / cpi0) ** (1 / n_years) - 1) * 100

        real_e0 = e0 / cpi0
        real_e1 = e1 / cpi1
        if real_e0 > 0 and real_e1 > 0:
            real_eps_growth = ((real_e1 / real_e0) ** (1 / n_years) - 1) * 100
        else:
            real_eps_growth = 0.0

        total_price_ret = ((p1 / p0) ** (1 / n_years) - 1) * 100
        total_nominal   = total_price_ret + div_yield
        multiple_change = total_nominal - div_yield - real_eps_growth - inflation

        rows.append({
            "regime":          name,
            "Dividend yield":  div_yield,
            "Real EPS growth": real_eps_growth,
            "Inflation":       inflation,
            "Multiple change": multiple_change,
            "Total":           total_nominal,
        })

    decomp = pd.DataFrame(rows).set_index("regime")

    components  = ["Dividend yield", "Real EPS growth", "Inflation", "Multiple change"]
    comp_colors = [
        COLORS["yield"], COLORS["growth"],
        COLORS["inflation"], COLORS["multiple"],
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    x      = np.arange(len(decomp))
    bottom = np.zeros(len(decomp))

    for comp, color in zip(components, comp_colors):
        vals = decomp[comp].values.astype(float)
        bars = ax.bar(
            x, vals, bottom=bottom, color=color,
            label=comp, width=0.55,
            edgecolor="white", linewidth=0.5,
        )
        for rect, val, bot in zip(bars, vals, bottom):
            if abs(val) > 0.4:
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    bot + val / 2,
                    f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold",
                )
        bottom = bottom + vals

    ax.plot(
        x, decomp["Total"].values.astype(float),
        "D", color=COLORS["total"],
        markersize=7, zorder=5, label="Total nominal return",
    )

    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(decomp.index, fontsize=9)
    ax.set_ylabel("Annualized contribution (percentage points)", labelpad=8)
    ax.set_title(
        "Decomposition of U.S. equity returns by regime\n"
        "S&P 500, annualized nominal return",
        pad=12,
    )
    ax.legend(fontsize=8, framealpha=0.5, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_return_decomposition.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_return_decomposition.png")


# =========================================================
# Chart 3 — 2022 repricing: real yields vs growth equity
# =========================================================
def chart_2022_repricing() -> None:
    import pandas_datareader.data as web
    from datetime import datetime

    start = datetime(2021, 1, 1)
    end   = datetime(2022, 12, 31)

    tickers = ["QQQ", "SPY"]
    prices  = {}
    for t in tickers:
        raw = yf.download(
            t, start=start, end=end,
            progress=False, auto_adjust=True,
        )
        if raw.empty:
            print(f"  Warning: no data for {t}")
            continue
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        prices[t] = close

    if len(prices) < 2:
        print("  Skipping 2022 repricing chart — equity download failed.")
        return

    try:
        real_yield = web.DataReader("DFII10", "fred", start, end)["DFII10"]
    except Exception as e:
        print(f"  Warning: FRED real yield download failed: {e}")
        return

    equity     = pd.DataFrame(prices).dropna()
    equity     = equity.reindex(real_yield.index).ffill().dropna()
    equity_idx = equity / equity.iloc[0] * 100

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(
        equity_idx.index, equity_idx["QQQ"],
        color=COLORS["equity"], lw=1.8, label="QQQ (Nasdaq-100)",
    )
    ax1.plot(
        equity_idx.index, equity_idx["SPY"],
        color=COLORS["neutral"], lw=1.4,
        linestyle="--", label="SPY (S&P 500)",
    )
    ax2.plot(
        real_yield.index, real_yield.values,
        color=COLORS["real_rate"], lw=1.8,
        label="10-yr TIPS real yield (%)",
    )

    ax1.set_ylabel(
        "Equity index (Jan 2021 = 100)",
        color=COLORS["equity"], labelpad=8,
    )
    ax2.set_ylabel(
        "10-year TIPS real yield (%)",
        color=COLORS["real_rate"], labelpad=8,
    )
    ax1.tick_params(axis="y", labelcolor=COLORS["equity"])
    ax2.tick_params(axis="y", labelcolor=COLORS["real_rate"])

    ax1.axvline(
        pd.Timestamp("2022-01-01"),
        color="#bbbbbb", lw=0.8, linestyle=":",
    )
    y_annot = float(equity_idx["QQQ"].min()) * 1.02
    ax1.text(
        pd.Timestamp("2022-01-15"), y_annot,
        "2022 begins", fontsize=7.5, color="#888888",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        fontsize=8, framealpha=0.5,
    )
    ax1.set_title(
        "Rising real yields and the repricing of long-duration equities\n"
        "January 2021 – December 2022",
        pad=12,
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_2022_repricing.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_2022_repricing.png")


# =========================================================
# Chart 4 — Valuation trap: European banks post-2011
# =========================================================
def chart_valuation_trap() -> None:
    start = "2011-01-01"
    end   = "2016-12-31"

    tickers = {
        "EUFN": "European Banks (EUFN)",
        "SPY":  "S&P 500 (SPY)",
    }
    prices = {}

    for t, label in tickers.items():
        raw = yf.download(
            t, start=start, end=end,
            progress=False, auto_adjust=True,
        )
        if raw.empty:
            print(f"  Warning: no data for {t}")
            continue
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        prices[label] = close

    if len(prices) < 2:
        print("  Skipping valuation trap chart — download failed.")
        return

    df_eq  = pd.DataFrame(prices).dropna()
    df_idx = df_eq / df_eq.iloc[0] * 100

    fig, ax = plt.subplots(figsize=(9, 4.5))

    col_colors = [COLORS["real_rate"], COLORS["neutral"]]
    for col, color in zip(df_idx.columns, col_colors):
        ax.plot(
            df_idx.index, df_idx[col],
            lw=1.8, color=color, label=col,
        )

    ax.axhline(100, color="#cccccc", lw=0.8, linestyle="--")

    eufn_col = "European Banks (EUFN)"
    if eufn_col in df_idx.columns:
        annot_date = pd.Timestamp("2016-01-01")
        idx_dates  = df_idx.index
        nearest    = idx_dates[idx_dates.get_indexer(
            [annot_date], method="nearest"
        )[0]]
        y_val = float(df_idx.loc[nearest, eufn_col])

        ax.annotate(
            "European banks remain below\n2011 levels through 2016\n"
            "despite low P/B ratios",
            xy=(nearest, y_val),
            xytext=(pd.Timestamp("2013-01-01"), 60),
            fontsize=8,
            arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
            color="#333333",
        )

    ax.set_ylabel("Total return index (Jan 2011 = 100)", labelpad=8)
    ax.set_title(
        "Valuation trap: European banks vs S&P 500, 2011–2016\n"
        "Low P/B ratios did not prevent sustained underperformance",
        pad=12,
    )
    ax.legend(fontsize=9, framealpha=0.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_valuation_trap.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_valuation_trap.png")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    print("Loading Shiller data...")
    shiller = load_shiller()
    print(f"  Loaded {len(shiller)} monthly observations "
          f"({shiller.index[0].year}–{shiller.index[-1].year})")

    print("\nBuilding Chart 1 — CAPE vs subsequent 10-year returns...")
    chart_cape_vs_returns(shiller)

    print("\nBuilding Chart 2 — Return decomposition by regime...")
    chart_return_decomposition(shiller)

    print("\nBuilding Chart 3 — 2022 real yield repricing...")
    chart_2022_repricing()

    print("\nBuilding Chart 4 — Valuation trap (European banks)...")
    chart_valuation_trap()

    print("\nAll charts saved to:", OUT_DIR.resolve())