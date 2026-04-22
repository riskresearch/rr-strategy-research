import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf
import pandas_datareader.data as web
from pathlib import Path
from datetime import datetime

# =========================================================
# Paths
# =========================================================
OUT_DIR = Path(__file__).parent.parent / "outputs" / "ch02"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    "nasdaq":    "#2166ac",
    "financials":"#d6604d",
    "sp500":     "#888888",
    "ipo":       "#4dac26",
    "price":     "#2166ac",
    "cape":      "#d6604d",
    "mortgage":  "#2166ac",
    "hpa":       "#4dac26",
    "default":   "#d6604d",
    "arkk":      "#d6604d",
    "qqq":       "#2166ac",
}


# =========================================================
# Download helper
# =========================================================
def fred(series: str, start: datetime, end: datetime) -> pd.Series:
    return web.DataReader(series, "fred", start, end).squeeze()


def yf_close(ticker: str, start: str, end: str) -> pd.Series:
    raw = yf.download(ticker, start=start, end=end,
                      progress=False, auto_adjust=True)
    if raw.empty:
        return pd.Series(dtype=float)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    return close.rename(ticker)


# =========================================================
# Chart 4 — Mortgage origination and home prices vs defaults
# =========================================================
def chart_mortgage_cycle() -> None:
    """
    Three-panel chart:
    Left:  MBA mortgage applications index (weekly -> annual avg)
           and Case-Shiller home price index
    Right: Delinquency rate on single-family residential mortgages
    Source: FRED
    """
    start = datetime(2000, 1, 1)
    end   = datetime(2010, 12, 31)

    print("  Downloading mortgage cycle data from FRED...")
    try:
        # MBA mortgage applications (weekly, FRED: MORTGAGE30US not ideal
        # use MSPUS - median sales price, CSUSHPISA - Case Shiller national)
        cs_hpi   = fred("CSUSHPISA",  start, end)   # Case-Shiller national HPI
        delinq   = fred("DRSFRMACBS", start, end)   # Delinquency rate SF mortgages
        # Mortgage originations approximation via new privately owned housing units
        housing  = fred("HOUST",      start, end)   # Housing starts (thousands)
    except Exception as e:
        print(f"  Warning: FRED download failed: {e}")
        return

    # Resample to quarterly for alignment
    cs_q     = cs_hpi.resample("QS").last().dropna()
    delinq_q = delinq.resample("QS").last().dropna()
    housing_q = housing.resample("QS").mean().dropna()

    # Align all series
    common = cs_q.index.intersection(delinq_q.index).intersection(housing_q.index)
    cs_q      = cs_q.loc[common]
    delinq_q  = delinq_q.loc[common]
    housing_q = housing_q.loc[common]

    # Rebase HPI to 100 at start
    cs_idx = cs_q / float(cs_q.iloc[0]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: housing starts and home price index
    ax  = axes[0]
    ax2 = ax.twinx()

    ax.fill_between(housing_q.index, housing_q.values,
                    color=COLORS["mortgage"], alpha=0.25)
    ax.plot(housing_q.index, housing_q.values,
            color=COLORS["mortgage"], lw=1.6,
            label="Housing starts (000s, left)")

    ax2.plot(cs_idx.index, cs_idx.values,
             color=COLORS["hpa"], lw=1.8, linestyle="--",
             label="Case-Shiller HPI (right, rebased)")

    # Annotate peak
    peak_date = cs_idx.idxmax()
    peak_val  = float(cs_idx.max())
    ax2.annotate(
        f"HPI peak\n{peak_date.strftime('%b %Y')}",
        xy=(peak_date, peak_val),
        xytext=(peak_date - pd.DateOffset(years=2), peak_val - 15),
        fontsize=8,
        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
        color="#333333",
    )

    ax.set_ylabel("Housing starts (thousands)", color=COLORS["mortgage"], labelpad=8)
    ax2.set_ylabel("Case-Shiller HPI (Q1 2000 = 100)", color=COLORS["hpa"], labelpad=8)
    ax.tick_params(axis="y", labelcolor=COLORS["mortgage"])
    ax2.tick_params(axis="y", labelcolor=COLORS["hpa"])

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, framealpha=0.5)
    ax.set_title("Housing activity and home prices\n2000–2010", pad=10)

    # Right: delinquency rate
    ax3 = axes[1]
    ax3.fill_between(delinq_q.index, delinq_q.values,
                     color=COLORS["default"], alpha=0.3)
    ax3.plot(delinq_q.index, delinq_q.values,
             color=COLORS["default"], lw=1.8,
             label="Delinquency rate (%)")

    # Annotate crisis onset
    crisis_date = pd.Timestamp("2007-09-01")
    idx = delinq_q.index.get_indexer([crisis_date], method="nearest")[0]
    if idx >= 0:
        ax3.axvline(delinq_q.index[idx], color="#cccccc", lw=0.8, linestyle=":")
        ax3.text(delinq_q.index[idx], float(delinq_q.max()) * 0.6,
                 "Crisis onset\n2007", fontsize=8, color="#555555")

    ax3.set_ylabel("Delinquency rate on SF mortgages (%)", labelpad=8)
    ax3.set_title("Mortgage delinquency rate\n2000–2010", pad=10)
    ax3.legend(fontsize=8, framealpha=0.5)

    fig.suptitle(
        "The U.S. housing cycle, 2000–2010\n"
        "Housing activity and prices peaked in 2005–2006; "
        "delinquencies surged from 2007. Source: FRED.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_mortgage_cycle.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_mortgage_cycle.png")


# =========================================================
# Chart 5 — Quality spread proxy: ARKK vs QQQ 2020–2022
# =========================================================
def chart_quality_spread() -> None:
    """
    ARK Innovation ETF (ARKK) as proxy for speculative/unprofitable
    growth versus Nasdaq-100 (QQQ) as proxy for quality large-cap tech.
    Shows relative performance narrowing in the boom and reversing sharply.
    """
    start = "2019-01-01"
    end   = "2022-12-31"

    arkk = yf_close("ARKK", start, end)
    qqq  = yf_close("QQQ",  start, end)

    if arkk.empty or qqq.empty:
        print("  Skipping quality spread chart — download failed.")
        return

    df     = pd.concat([arkk, qqq], axis=1).dropna()
    df_idx = df / df.iloc[0] * 100

    # Relative: ARKK / QQQ
    rel = df_idx["ARKK"] / df_idx["QQQ"] * 100

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Top: absolute performance
    ax = axes[0]
    ax.plot(df_idx.index, df_idx["ARKK"],
            color=COLORS["arkk"], lw=1.8,
            label="ARK Innovation (ARKK) — speculative growth proxy")
    ax.plot(df_idx.index, df_idx["QQQ"],
            color=COLORS["qqq"], lw=1.8,
            label="Nasdaq-100 (QQQ) — quality large-cap proxy")
    ax.set_ylabel("Total return index (Jan 2019 = 100)", labelpad=8)
    ax.set_title(
        "Speculative vs quality growth: ARKK vs QQQ, 2019–2022\n"
        "Quality dispersion narrows during the boom and snaps back sharply",
        pad=12,
    )
    ax.legend(fontsize=8, framealpha=0.5)

    # Annotate ARKK peak
    peak_date = df_idx["ARKK"].idxmax()
    peak_val  = float(df_idx["ARKK"].max())
    ax.annotate(
        f"ARKK peak\n{peak_date.strftime('%b %Y')}\n+{peak_val-100:.0f}%",
        xy=(peak_date, peak_val),
        xytext=(peak_date - pd.DateOffset(months=8), peak_val - 80),
        fontsize=8,
        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
        color="#333333",
    )

    # Bottom: relative performance
    ax2 = axes[1]
    ax2.plot(rel.index, rel.values,
             color=COLORS["arkk"], lw=1.8,
             label="ARKK / QQQ relative index")
    ax2.axhline(100, color="#cccccc", lw=0.8, linestyle="--")
    ax2.fill_between(rel.index, rel.values, 100,
                     where=(rel.values > 100),
                     alpha=0.2, color=COLORS["arkk"],
                     label="ARKK outperforms (quality spread narrows)")
    ax2.fill_between(rel.index, rel.values, 100,
                     where=(rel.values <= 100),
                     alpha=0.2, color=COLORS["qqq"],
                     label="ARKK underperforms (quality spread widens)")

    trough_date = rel.idxmin()
    trough_val  = float(rel.min())
    ax2.annotate(
        f"ARKK / QQQ trough\n{trough_date.strftime('%b %Y')}\n"
        f"Ratio = {trough_val:.0f}",
        xy=(trough_date, trough_val),
        xytext=(trough_date - pd.DateOffset(months=6), trough_val + 20),
        fontsize=8,
        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
        color="#333333",
    )

    ax2.set_ylabel("ARKK / QQQ relative index (Jan 2019 = 100)", labelpad=8)
    ax2.legend(fontsize=8, framealpha=0.5, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_quality_spread.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_quality_spread.png")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    print("Building Chart 4 — Mortgage cycle...")
    chart_mortgage_cycle()

    print("\nBuilding Chart 5 — Quality spread proxy ARKK vs QQQ...")
    chart_quality_spread()

    print("\nNew charts saved to:", OUT_DIR.resolve())