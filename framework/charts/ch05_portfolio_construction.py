import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import pandas_datareader.data as web
import urllib.request
import zipfile
import io
from datetime import datetime

# =========================================================
# Paths
# =========================================================
OUT_DIR = Path(__file__).parent.parent / "outputs" / "ch05"
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
    "equity":   "#2166ac",
    "bond":     "#4dac26",
    "calm":     "#2166ac",
    "stress":   "#d6604d",
    "neutral":  "#888888",
    "rebal":    "#2166ac",
    "norebal":  "#d6604d",
    "value":    "#d6604d",
    "momentum": "#2166ac",
    "combined": "#4dac26",
}


# =========================================================
# Download helpers
# =========================================================
def fred(series: str, start: datetime, end: datetime) -> pd.Series:
    return web.DataReader(series, "fred", start, end).squeeze()


def download_ff_zip(url: str) -> list:
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        fname = [n for n in z.namelist()
                 if n.endswith(".CSV") or n.endswith(".csv")][0]
        with z.open(fname) as f:
            lines = f.read().decode(
                "utf-8", errors="replace"
            ).splitlines()

    data_lines = []
    started    = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
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

    records = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            date_int = int(parts[0])
            year     = date_int // 100
            month    = date_int % 100
            if 1 <= month <= 12 and 1900 <= year <= 2100:
                vals = [
                    float(p) if p not in ["", "."] else np.nan
                    for p in parts[1:]
                ]
                records.append(
                    [pd.Timestamp(year=year, month=month, day=1)]
                    + vals
                )
        except (ValueError, TypeError):
            continue
    return records


# =========================================================
# Chart 1 — Stress vs calm correlation comparison
# =========================================================
def chart_stress_correlations() -> None:
    """
    Compare correlations between equity and credit in:
    - Calm periods (bottom 75% of equity volatility months)
    - Stress periods (top 25% of equity volatility months)
    Use FF market factor and HY spread changes as proxies.
    Also show equity-bond and equity-equity (domestic/intl).
    """
    ff3_url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Research_Data_Factors_CSV.zip"
    )
    ff_intl_url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_International_Countries_3_Factors_CSV.zip"
    )

    print("  Downloading FF factors for stress correlation...")
    try:
        ff3_records = download_ff_zip(ff3_url)
    except Exception as e:
        print(f"  Warning: FF3 download failed: {e}")
        return

    ff3 = pd.DataFrame(
        ff3_records,
        columns=["date", "MktRF", "SMB", "HML", "RF"],
    ).set_index("date").sort_index()
    for col in ff3.columns:
        ff3[col] = pd.to_numeric(ff3[col], errors="coerce")
    ff3 = ff3 / 100

    # Use market return and bond return proxy
    start = datetime(1990, 1, 1)
    end   = datetime(2023, 12, 31)
    try:
        gs10 = fred("GS10", start, end)
        spx_vix = fred("VIXCLS", start, end)
    except Exception as e:
        print(f"  Warning: FRED download failed: {e}")
        return

    gs10_m   = gs10.resample("MS").last().dropna()
    bond_ret = (
        -7.0 * gs10_m.diff() / 100 + gs10_m / 1200
    ).dropna()
    bond_ret.index = pd.DatetimeIndex([
        pd.Timestamp(year=d.year, month=d.month, day=1)
        for d in bond_ret.index
    ])

    eq_ret = (ff3["MktRF"] + ff3["RF"]).loc[
        start.strftime("%Y-%m-%d"):
        end.strftime("%Y-%m-%d")
    ].dropna()

    common    = eq_ret.index.intersection(bond_ret.index)
    eq_ret_c  = eq_ret.loc[common]
    bond_ret_c = bond_ret.loc[common]

    # Define stress periods using rolling equity volatility
    eq_vol = eq_ret_c.rolling(3, min_periods=2).std()
    stress_threshold = eq_vol.quantile(0.75)
    stress_mask = eq_vol >= stress_threshold
    calm_mask   = eq_vol < stress_threshold

    # HY spread changes as credit proxy
    hy_annual = {
        1997: 310, 1998: 480, 1999: 570, 2000: 735,
        2001: 870, 2002: 920, 2003: 580, 2004: 340,
        2005: 310, 2006: 280, 2007: 470, 2008: 1490,
        2009: 870, 2010: 530, 2011: 620, 2012: 520,
        2013: 400, 2014: 430, 2015: 610, 2016: 540,
        2017: 360, 2018: 440, 2019: 400, 2020: 590,
        2021: 300, 2022: 480, 2023: 390,
    }
    hy_s = pd.Series(
        {pd.Timestamp(year=y, month=1, day=1): v
         for y, v in hy_annual.items()}
    ).resample("MS").ffill()
    hy_ret = (-hy_s.diff() / 10000 * 4).dropna()
    hy_ret.index = pd.DatetimeIndex([
        pd.Timestamp(year=d.year, month=d.month, day=1)
        for d in hy_ret.index
    ])

    common2 = eq_ret_c.index.intersection(hy_ret.index)
    eq_c    = eq_ret_c.loc[common2]
    hy_c    = hy_ret.loc[common2]
    stress2 = stress_mask.reindex(common2).fillna(False)
    calm2   = calm_mask.reindex(common2).fillna(False)

    pairs = [
        ("Equity vs Bonds",
         eq_ret_c, bond_ret_c, stress_mask, calm_mask),
        ("Equity vs HY Credit",
         eq_c, hy_c, stress2, calm2),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (title, s1, s2, st_mask, cl_mask) in zip(axes, pairs):
        common_idx = s1.index.intersection(s2.index)
        s1_a = s1.reindex(common_idx)
        s2_a = s2.reindex(common_idx)
        st   = st_mask.reindex(common_idx).fillna(False)
        cl   = cl_mask.reindex(common_idx).fillna(False)

        corr_calm   = float(s1_a[cl].corr(s2_a[cl]))
        corr_stress = float(s1_a[st].corr(s2_a[st]))
        corr_full   = float(s1_a.corr(s2_a))

        labels = [
            f"Full sample\n(n={len(s1_a)})",
            f"Calm periods\n(n={cl.sum()})",
            f"Stress periods\n(n={st.sum()})",
        ]
        values = [corr_full, corr_calm, corr_stress]
        colors = [
            COLORS["neutral"],
            COLORS["calm"],
            COLORS["stress"],
        ]

        bars = ax.bar(labels, values, color=colors,
                      width=0.5, edgecolor="white")
        for bar, val in zip(bars, values):
            ypos = val + 0.02 if val >= 0 else val - 0.05
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ypos,
                f"{val:.2f}",
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=9, fontweight="bold",
            )
        ax.axhline(0, color="#999999", lw=0.8, linestyle="--")
        ax.set_ylim(-0.8, 1.0)
        ax.set_title(title, pad=10)
        ax.set_ylabel("Pearson correlation", labelpad=8)

    fig.suptitle(
        "Correlations in calm versus stress periods, 1990–2023\n"
        "Stress = top 25% of rolling 3-month equity volatility. "
        "Source: Ken French Data Library, FRED.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        OUT_DIR / "fig_stress_correlations.png",
        bbox_inches="tight",
    )
    plt.close(fig)
    print("  Saved: fig_stress_correlations.png")


# =========================================================
# Chart 2 — Position sizing table
# =========================================================
def chart_position_sizing_table() -> None:
    """
    Show the same investment thesis expressed at three position sizes.
    Columns: size, portfolio vol contribution, max tolerable loss,
    required hit rate to be net positive in expectation.
    Assumptions: portfolio vol = 12%, thesis vol = 25%,
    corr with portfolio = 0.3, win/loss ratio = 1.5.
    """
    portfolio_vol  = 0.12
    thesis_vol     = 0.25
    corr           = 0.30
    win_loss_ratio = 1.5    # avg win / avg loss magnitude

    sizes = [0.02, 0.08, 0.15]

    rows = []
    for w in sizes:
        # Marginal vol contribution (approximate)
        marginal_vol = w * (
            w * thesis_vol ** 2
            + (1 - w) * corr * thesis_vol * portfolio_vol
        ) / portfolio_vol
        port_vol_contrib = marginal_vol * 100

        # Max tolerable loss: position loss that moves portfolio
        # down by 2 * portfolio_vol (approx 2-sigma event)
        max_port_loss = 2 * portfolio_vol
        max_pos_loss  = max_port_loss / w
        max_pos_loss_pct = max_pos_loss * 100

        # Required hit rate for positive expectation
        # E[return] = p * win_loss_ratio * L - (1-p) * L > 0
        # p > 1 / (1 + win_loss_ratio)
        req_hit_rate = 1 / (1 + win_loss_ratio) * 100

        rows.append([
            f"{w*100:.0f}%",
            f"{port_vol_contrib:.2f}%",
            f"{max_pos_loss_pct:.0f}%",
            f"{req_hit_rate:.1f}%",
        ])

    col_labels = [
        "Position size\n(% of portfolio)",
        "Portfolio vol\ncontribution",
        "Position loss that\nhits 2σ portfolio level",
        "Required hit rate\n(win/loss = 1.5x)",
    ]

    fig, ax = plt.subplots(figsize=(10, 2.8), facecolor="white")
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.18, 0.22, 0.32, 0.28],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.5)

    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#2166ac")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    row_colors = ["#d1e8f7", "#d9f0d3", "#fde8d8"]
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            cell = tbl[i, j]
            cell.set_facecolor(row_colors[i - 1])
            cell.set_edgecolor("#eeeeee")

    ax.set_title(
        "The same thesis expressed at three position sizes\n"
        "Portfolio vol = 12%, Thesis vol = 25%, "
        "Corr with portfolio = 0.30, Win/loss ratio = 1.5x",
        fontsize=9, pad=15, loc="left",
    )

    fig.tight_layout()
    fig.savefig(
        OUT_DIR / "fig_position_sizing_table.png",
        bbox_inches="tight", dpi=180,
    )
    plt.close(fig)
    print("  Saved: fig_position_sizing_table.png")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    print("Building Chart — Stress vs calm correlations...")
    chart_stress_correlations()

    print("\nBuilding Chart — Position sizing table...")
    chart_position_sizing_table()

    print("\nCharts saved to:", OUT_DIR.resolve())