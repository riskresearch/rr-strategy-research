import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import urllib.request
import zipfile
import io

# =========================================================
# Paths
# =========================================================
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = Path(__file__).parent.parent / "outputs" / "ch03"
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
    "underperform": "#d6604d",
    "outperform":   "#2166ac",
    "insample":     "#2166ac",
    "oosample":     "#756bb1",
    "postpub":      "#d6604d",
    "high_inf":     "#d6604d",
    "low_inf":      "#2166ac",
    "neutral":      "#888888",
}


# =========================================================
# Chart 1 — SPIVA scorecard bar chart
# =========================================================
def chart_spiva() -> None:
    """
    Hardcoded from S&P SPIVA U.S. Year-End 2023 Scorecard.
    Percentage of active large-cap equity funds underperforming
    the S&P 500 over 1, 5, 10, and 15 years.
    Source: S&P Dow Jones Indices, SPIVA U.S. Year-End 2023.
    """
    horizons      = ["1 Year", "5 Years", "10 Years", "15 Years"]
    pct_under     = [60.0, 78.5, 87.7, 88.4]
    pct_over      = [100 - p for p in pct_under]

    x      = np.arange(len(horizons))
    width  = 0.55

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_under = ax.bar(
        x, pct_under, width,
        color=COLORS["underperform"],
        label="Underperform benchmark",
        edgecolor="white", linewidth=0.5,
    )
    bars_over = ax.bar(
        x, pct_over, width,
        bottom=pct_under,
        color=COLORS["outperform"],
        alpha=0.6,
        label="Outperform benchmark",
        edgecolor="white", linewidth=0.5,
    )

    # Label underperformance percentages
    for bar, val in zip(bars_under, pct_under):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val / 2,
            f"{val:.1f}%",
            ha="center", va="center",
            fontsize=9, color="white", fontweight="bold",
        )

    ax.axhline(50, color="#999999", lw=0.8, linestyle="--", alpha=0.7)
    ax.text(
        len(horizons) - 0.45, 51.5,
        "50% line", fontsize=7.5, color="#999999",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=10)
    ax.set_ylabel("Percentage of funds (%)", labelpad=8)
    ax.set_ylim(0, 105)
    ax.set_title(
        "Most active large-cap equity funds underperform over time\n"
        "S&P 500 benchmark, net of fees — SPIVA U.S. Year-End 2023",
        pad=12,
    )
    ax.legend(fontsize=9, framealpha=0.5, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_spiva.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_spiva.png")


# =========================================================
# Chart 2 — McLean and Pontiff anomaly decay
# =========================================================
def chart_anomaly_decay() -> None:
    """
    Hardcoded from McLean and Pontiff (2016), Table 1 and Figure 1.
    Average monthly returns for 97 anomalies across three periods:
      - In-sample (full sample used in original paper)
      - Out-of-sample (post-sample-end, pre-publication)
      - Post-publication
    Source: McLean & Pontiff, Journal of Finance 71(1), 2016.
    """
    periods   = ["In-sample", "Out-of-sample\n(pre-publication)", "Post-publication"]
    # Average monthly returns in percent, from paper Table 1
    returns   = [0.58, 0.42, 0.26]
    # Decay relative to in-sample
    decay_pct = [0, (0.58 - 0.42) / 0.58 * 100, (0.58 - 0.26) / 0.58 * 100]

    colors = [
        COLORS["insample"],
        COLORS["oosample"],
        COLORS["postpub"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left panel — average returns
    ax = axes[0]
    bars = ax.bar(
        periods, returns,
        color=colors, width=0.5,
        edgecolor="white", linewidth=0.5,
    )
    for bar, val in zip(bars, returns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.2f}%",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )
    ax.set_ylabel("Average monthly return (%)", labelpad=8)
    ax.set_title("Average anomaly return\nacross three periods", pad=10)
    ax.set_ylim(0, 0.75)

    # Right panel — cumulative decay
    ax2 = axes[1]
    bars2 = ax2.bar(
        periods, decay_pct,
        color=colors, width=0.5,
        edgecolor="white", linewidth=0.5,
    )
    for bar, val in zip(bars2, decay_pct):
        if val > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.5,
                f"{val:.0f}%",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
            )
    ax2.set_ylabel("Decay from in-sample return (%)", labelpad=8)
    ax2.set_title("Cumulative decay from\nin-sample benchmark", pad=10)
    ax2.set_ylim(0, 70)

    fig.suptitle(
        "Academic anomalies decay after discovery\n"
        "97 return predictors, McLean and Pontiff (2016)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_anomaly_decay.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_anomaly_decay.png")


# =========================================================
# Chart 3 — Momentum across inflation regimes
# =========================================================
def chart_momentum_regimes() -> None:
    """
    Download Ken French momentum factor (Mom) from his data library.
    Download CPI from FRED via pandas_datareader.
    Split months into high/low inflation regimes and compare
    momentum factor average monthly return across regimes.
    """
    import pandas_datareader.data as web
    from datetime import datetime

    # --- Download French momentum factor ---
    mom_url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Momentum_Factor_CSV.zip"
    )
    print("  Downloading French momentum factor...")
    try:
        with urllib.request.urlopen(mom_url, timeout=30) as resp:
            raw_bytes = resp.read()
    except Exception as e:
        print(f"  Warning: momentum download failed: {e}")
        return

    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
        fname = [n for n in z.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        with z.open(fname) as f:
            lines = f.read().decode("utf-8", errors="replace").splitlines()

    # Find where the monthly data starts (skip header rows)
    data_lines = []
    started    = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Data rows start with a 6-digit YYYYMM date
        parts = [p.strip() for p in stripped.split(",")]
        if len(parts) >= 2:
            try:
                int(parts[0])
                if len(parts[0]) == 6:
                    started = True
            except ValueError:
                if started:
                    break
                continue
        if started:
            data_lines.append(stripped)

    if not data_lines:
        print("  Warning: could not parse momentum factor data.")
        return

    records = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            date_int = int(parts[0])
            mom_val  = float(parts[1])
            year     = date_int // 100
            month    = date_int % 100
            if 1 <= month <= 12 and 1900 <= year <= 2100:
                records.append({
                    "date": pd.Timestamp(year=year, month=month, day=1),
                    "mom":  mom_val,
                })
        except (ValueError, TypeError):
            continue

    mom = pd.DataFrame(records).set_index("date")["mom"].sort_index()
    mom = mom[mom.index >= "1930-01-01"]

    # --- Download CPI from FRED ---
    print("  Downloading CPI from FRED...")
    try:
        cpi = web.DataReader(
            "CPIAUCSL", "fred",
            datetime(1929, 1, 1), datetime(2023, 12, 31),
        )["CPIAUCSL"]
    except Exception as e:
        print(f"  Warning: CPI download failed: {e}")
        return

    cpi_m      = cpi.resample("MS").last()
    cpi_yoy    = cpi_m.pct_change(12) * 100
    cpi_yoy    = cpi_yoy.dropna()
    cpi_yoy.index = pd.DatetimeIndex([
    pd.Timestamp(year=d.year, month=d.month, day=1)
    for d in cpi_yoy.index])
    # Align
    common = mom.index.intersection(cpi_yoy.index)
    mom    = mom.loc[common]
    cpi_yoy = cpi_yoy.loc[common]

    # Define high/low inflation regime using median split
    median_inf = float(cpi_yoy.median())

    high_inf_mask = cpi_yoy >= median_inf
    low_inf_mask  = cpi_yoy <  median_inf

    mom_high = mom[high_inf_mask]
    mom_low  = mom[low_inf_mask]

    # Bootstrap confidence intervals
    rng    = np.random.default_rng(42)
    n_boot = 2000

    def boot_mean(series, n):
        means = [
            series.sample(frac=1, replace=True, random_state=rng.integers(1e6)).mean()
            for _ in range(n)
        ]
        return np.percentile(means, [2.5, 97.5])

    mean_high = float(mom_high.mean())
    mean_low  = float(mom_low.mean())
    ci_high   = boot_mean(mom_high, n_boot)
    ci_low    = boot_mean(mom_low,  n_boot)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: bar comparison
    ax = axes[0]
    labels = [
        f"Low inflation\n(<{median_inf:.1f}% YoY)",
        f"High inflation\n(≥{median_inf:.1f}% YoY)",
    ]
    means  = [mean_low, mean_high]
    cis    = [ci_low,   ci_high]
    colors = [COLORS["low_inf"], COLORS["high_inf"]]

    for i, (label, mean, ci, color) in enumerate(zip(labels, means, cis, colors)):
        ax.bar(i, mean, color=color, width=0.45,
               edgecolor="white", linewidth=0.5, label=label)
        ax.errorbar(
            i, mean,
            yerr=[[mean - ci[0]], [ci[1] - mean]],
            fmt="none", color="#333333", capsize=5, lw=1.5,
        )
        ax.text(
            i, mean + 0.02,
            f"{mean:.2f}%/mo",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )

    ax.axhline(0, color="#999999", lw=0.8, linestyle="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Average monthly return (%)", labelpad=8)
    ax.set_title(
        "Momentum factor return\nby inflation regime",
        pad=10,
    )

    # Right: rolling 5-year momentum return with inflation overlay
    ax2    = axes[1]
    ax2b   = ax2.twinx()

    roll_mom = mom.rolling(60, min_periods=48).mean()
    roll_cpi = cpi_yoy.rolling(60, min_periods=48).mean()

    ax2.plot(
        roll_mom.index, roll_mom.values,
        color=COLORS["low_inf"], lw=1.6,
        label="5-yr rolling avg momentum return",
    )
    ax2b.plot(
        roll_cpi.index, roll_cpi.values,
        color=COLORS["high_inf"], lw=1.2,
        linestyle="--", alpha=0.7,
        label="5-yr rolling avg CPI YoY (%)",
    )

    ax2.axhline(0, color="#cccccc", lw=0.8)
    ax2.set_ylabel(
        "Rolling avg momentum return (%/mo)",
        color=COLORS["low_inf"], labelpad=8,
    )
    ax2b.set_ylabel(
        "Rolling avg CPI YoY (%)",
        color=COLORS["high_inf"], labelpad=8,
    )
    ax2.tick_params(axis="y", labelcolor=COLORS["low_inf"])
    ax2b.tick_params(axis="y", labelcolor=COLORS["high_inf"])

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(
        lines1 + lines2, labels1 + labels2,
        fontsize=7.5, framealpha=0.5,
    )
    ax2.set_title(
        "Rolling momentum return vs inflation\n1930–2023",
        pad=10,
    )

    fig.suptitle(
        "Momentum factor performance across inflation regimes\n"
        "Ken French momentum factor, monthly, 1930–2023",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_momentum_regimes.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_momentum_regimes.png")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    print("Building Chart 1 — SPIVA scorecard...")
    chart_spiva()

    print("\nBuilding Chart 2 — McLean and Pontiff anomaly decay...")
    chart_anomaly_decay()

    print("\nBuilding Chart 3 — Momentum across inflation regimes...")
    chart_momentum_regimes()

    print("\nAll charts saved to:", OUT_DIR.resolve())