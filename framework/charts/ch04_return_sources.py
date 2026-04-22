import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import pandas_datareader.data as web
import urllib.request
import zipfile
import io
import yfinance as yf
from datetime import datetime

# =========================================================
# Paths
# =========================================================
OUT_DIR = Path(__file__).parent.parent / "outputs" / "ch04"
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
    "positive":    "#2166ac",
    "negative":    "#d6604d",
    "neutral":     "#888888",
    "equity":      "#2166ac",
    "bond":        "#4dac26",
    "spread":      "#756bb1",
    "market":      "#1a1a1a",
    "hml":         "#d6604d",
    "umd":         "#2166ac",
    "smb":         "#4dac26",
    "alpha":       "#756bb1",
    "beta_bar":    "#2166ac",
    "factor_bar":  "#4dac26",
    "alpha_bar":   "#d6604d",
}


# =========================================================
# Helper: download and parse Fama-French zip
# =========================================================
def download_ff_zip(url: str) -> list:
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        fname = [n for n in z.namelist()
                 if n.endswith(".CSV") or n.endswith(".csv")][0]
        with z.open(fname) as f:
            lines = f.read().decode("utf-8", errors="replace").splitlines()

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
                vals = [float(p) if p not in ["", "."] else np.nan
                        for p in parts[1:]]
                records.append(
                    [pd.Timestamp(year=year, month=month, day=1)] + vals
                )
        except (ValueError, TypeError):
            continue
    return records


def normalize_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalize all dates to month-start."""
    return pd.DatetimeIndex([
        pd.Timestamp(year=d.year, month=d.month, day=1)
        for d in idx
    ])


# =========================================================
# Chart 1 — Rolling stock-bond correlation (full history)
# =========================================================
def chart_stock_bond_correlation() -> None:
    """
    Rolling 24-month stock-bond correlation using:
    - Equity: Fama-French market factor (MktRF + RF)
    - Bond: approximate monthly return from GS10 yield changes
    Full history back to 1952.
    """
    ff3_url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Research_Data_Factors_CSV.zip"
    )

    print("  Downloading FF market factor and GS10...")
    try:
        ff3_records = download_ff_zip(ff3_url)
        gs10 = web.DataReader(
            "GS10", "fred",
            datetime(1952, 1, 1),
            datetime(2024, 6, 30),
        ).squeeze()
    except Exception as e:
        print(f"  Warning: download failed: {e}")
        return

    ff3 = pd.DataFrame(
        ff3_records, columns=["date", "MktRF", "SMB", "HML", "RF"]
    ).set_index("date").sort_index()
    for col in ff3.columns:
        ff3[col] = pd.to_numeric(ff3[col], errors="coerce")
    ff3 = ff3 / 100

    eq_ret = (ff3["MktRF"] + ff3["RF"]).dropna()

    gs10_m   = gs10.resample("MS").last().dropna()
    bond_ret = (-7.0 * gs10_m.diff() / 100 + gs10_m / 1200).dropna()
    bond_ret.index = normalize_index(bond_ret.index)

    common   = eq_ret.index.intersection(bond_ret.index)
    eq_ret   = eq_ret.loc[common]
    bond_ret = bond_ret.loc[common]

    roll_corr = eq_ret.rolling(24, min_periods=20).corr(bond_ret)

    fig, ax1 = plt.subplots(figsize=(11, 5))

    ax1.fill_between(
        roll_corr.index, roll_corr.values, 0,
        where=(roll_corr.values > 0),
        alpha=0.25, color=COLORS["negative"],
        label="Positive correlation (diversification fails)",
    )
    ax1.fill_between(
        roll_corr.index, roll_corr.values, 0,
        where=(roll_corr.values <= 0),
        alpha=0.25, color=COLORS["positive"],
        label="Negative correlation (diversification works)",
    )
    ax1.plot(
        roll_corr.index, roll_corr.values,
        color="#333333", lw=1.4,
        label="24-month rolling correlation",
    )
    ax1.axhline(0, color="#999999", lw=1.0, linestyle="--")

    annotations = [
        ("1970s\ninflation",        "1975-01-01",  0.55),
        ("Disinflation\nbond bull", "1995-01-01", -0.60),
        ("2022\nrate shock",        "2022-06-01",  0.55),
    ]
    for label, date_str, y_pos in annotations:
        dt  = pd.Timestamp(date_str)
        idx = roll_corr.index.get_indexer([dt], method="nearest")[0]
        if idx >= 0:
            ax1.axvline(roll_corr.index[idx],
                        color="#cccccc", lw=0.8, linestyle=":")
            ax1.text(roll_corr.index[idx], y_pos, label,
                     fontsize=7.5, color="#555555", ha="center")

    ax1.set_ylabel("Rolling 24-month stock-bond correlation", labelpad=8)
    ax1.set_ylim(-1, 1)
    ax1.set_title(
        "U.S. stock-bond correlation, 1952–2024\n"
        "Equity (FF market factor) vs 10-year Treasury, "
        "rolling 24-month window",
        pad=12,
    )
    ax1.legend(fontsize=8, framealpha=0.5, loc="lower left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_stock_bond_correlation.png",
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_stock_bond_correlation.png")


# =========================================================
# Chart 2 — 60/40 risk contribution breakdown
# =========================================================
def chart_risk_contribution() -> None:
    """
    Illustrate that a 60/40 portfolio is dominated by equity risk
    across three correlation assumptions.
    Uses Euler risk contribution decomposition.
    Note: when correlation is negative, bonds have negative risk
    contribution — equity contribution exceeds 100% to compensate.
    This is mathematically correct, not an error.
    """
    vol_eq   = 0.16
    vol_bond = 0.06
    w_eq     = 0.60
    w_bond   = 0.40

    correlations = [-0.30, 0.00, 0.30]
    corr_labels  = [
        "Negative corr\n(ρ = -0.30)\nPost-2000 regime",
        "Zero corr\n(ρ = 0.00)",
        "Positive corr\n(ρ = +0.30)\n1970s / 2022 regime",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    for ax, rho, label in zip(axes, correlations, corr_labels):
        port_var = (
            (w_eq * vol_eq) ** 2
            + (w_bond * vol_bond) ** 2
            + 2 * w_eq * w_bond * rho * vol_eq * vol_bond
        )
        port_vol = np.sqrt(port_var)

        # Euler decomposition: RC_i = w_i * Cov(r_i, r_p) / sigma_p
        cov_eq_port   = w_eq * vol_eq ** 2 + w_bond * rho * vol_eq * vol_bond
        cov_bond_port = w_bond * vol_bond ** 2 + w_eq * rho * vol_eq * vol_bond
        rc_eq   = w_eq   * cov_eq_port   / port_vol
        rc_bond = w_bond * cov_bond_port / port_vol

        rc_eq_pct   = rc_eq   / port_vol * 100
        rc_bond_pct = rc_bond / port_vol * 100

        # For pie chart, clip to non-negative for display
        # but show true value in label
        sizes  = [max(rc_eq_pct, 0.1), max(abs(rc_bond_pct), 0.1)]
        clrs   = [COLORS["equity"], COLORS["bond"]]
        labels_pie = [
            f"Equity\n{rc_eq_pct:.0f}% of risk",
            f"Bonds\n{rc_bond_pct:.0f}% of risk",
        ]

        wedges, _ = ax.pie(
            sizes, colors=clrs,
            startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=1.5),
        )
        ax.set_title(label, fontsize=8.5, pad=10)

        for wedge, lbl in zip(wedges, labels_pie):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = 0.65 * np.cos(np.deg2rad(angle))
            y = 0.65 * np.sin(np.deg2rad(angle))
            ax.text(
                x, y, lbl,
                ha="center", va="center",
                fontsize=8, color="white", fontweight="bold",
            )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["equity"],
              label="Equity (60% capital weight)"),
        Patch(facecolor=COLORS["bond"],
              label="Bonds (40% capital weight)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=2,
        fontsize=9, framealpha=0.5,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.suptitle(
        "Risk contribution in a 60/40 portfolio across correlation regimes\n"
        "Euler decomposition. Equity vol = 16%, Bond vol = 6%.\n"
        "When correlation is negative, bonds reduce total risk — "
        "equity contribution can exceed 100%.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_risk_contribution.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_risk_contribution.png")


# =========================================================
# Chart 3 — High-yield spread vs forward return
# =========================================================
def chart_hy_spread_vs_return() -> None:
    """
    Annual average HY OAS hardcoded from published sources.
    Source: ICE BofA U.S. High Yield Index OAS as reported in
    FRED historical publications, Ilmanen (2011), JPMorgan HY research.
    """
    hy_annual = {
        1997: 310,  1998: 480,  1999: 570,  2000: 735,
        2001: 870,  2002: 920,  2003: 580,  2004: 340,
        2005: 310,  2006: 280,  2007: 470,  2008: 1490,
        2009: 870,  2010: 530,  2011: 620,  2012: 520,
        2013: 400,  2014: 430,  2015: 610,  2016: 540,
        2017: 360,  2018: 440,  2019: 400,  2020: 590,
        2021: 300,  2022: 480,  2023: 390,
    }

    years   = sorted(hy_annual.keys())
    spreads = [hy_annual[y] for y in years]

    fwd_rets  = []
    fwd_years = []
    for i, year in enumerate(years):
        if i + 3 >= len(years):
            break
        spread_start = spreads[i]
        spread_end   = spreads[i + 3]
        carry_ann    = spread_start / 100
        spread_chg   = (spread_end - spread_start) / 100
        duration     = 4.0
        price_drag   = spread_chg * duration / 3
        default_drag = spread_start * 0.35 / 100 / 3
        fwd_ret      = (carry_ann - price_drag - default_drag) * 100
        fwd_rets.append(fwd_ret)
        fwd_years.append(year)

    spread_for_scatter = [hy_annual[y] for y in fwd_years]
    x      = np.array(spread_for_scatter)
    y      = np.array(fwd_rets)
    coeffs = np.polyfit(x, y, 1)
    x_fit  = np.linspace(x.min(), x.max(), 200)
    y_fit  = np.polyval(coeffs, x_fit)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.scatter(
        spread_for_scatter, fwd_rets,
        color=COLORS["spread"], s=40, alpha=0.7, zorder=3,
        label="Annual observations",
    )
    ax.plot(x_fit, y_fit, color=COLORS["negative"], lw=1.8,
            label=f"OLS fit (slope={coeffs[0]:.3f})")
    ax.axhline(0, color="#999999", lw=0.8, linestyle="--")

    for label, year, offset in [
        ("2008\n(OAS≈1490bp)", 2008, ( 30, -2)),
        ("2006\n(OAS≈280bp)",  2006, ( 20,  3)),
        ("2020\n(OAS≈590bp)",  2020, ( 20,  3)),
    ]:
        if year in fwd_years:
            idx = fwd_years.index(year)
            ax.annotate(
                label,
                xy=(spread_for_scatter[idx], fwd_rets[idx]),
                xytext=(
                    spread_for_scatter[idx] + offset[0],
                    fwd_rets[idx] + offset[1],
                ),
                fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
                color="#333333",
            )

    ax.set_xlabel("HY OAS at start of year (bps)", labelpad=8)
    ax.set_ylabel("Approx. 3-year annualized forward return (%)",
                  labelpad=8)
    ax.set_title("Starting spread and forward HY return\n1997–2023",
                 pad=10)
    ax.legend(fontsize=8, framealpha=0.5)

    ax2 = axes[1]
    bar_colors = [
        COLORS["negative"] if s >= 700 else
        COLORS["positive"] if s <= 350 else
        COLORS["neutral"]
        for s in spreads
    ]
    ax2.bar(years, spreads, color=bar_colors,
            width=0.75, edgecolor="white")
    ax2.axhline(700, color=COLORS["negative"], lw=1.0,
                linestyle="--", alpha=0.8,
                label="700bp — historically wide")
    ax2.axhline(350, color=COLORS["positive"], lw=1.0,
                linestyle="--", alpha=0.8,
                label="350bp — historically tight")
    ax2.set_ylabel("Annual average HY OAS (bps)", labelpad=8)
    ax2.set_title("High-yield spread history\n1997–2023", pad=10)
    ax2.legend(fontsize=8, framealpha=0.5)

    fig.suptitle(
        "High-yield credit: starting spread as a forward return signal\n"
        "ICE BofA U.S. High Yield Index OAS — annual averages",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_hy_spread_returns.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_hy_spread_returns.png")


# =========================================================
# Chart 4 — Factor premia summary table
# =========================================================
def chart_factor_table() -> None:
    data = {
        "Factor": [
            "Equity premium",
            "Value (HML)",
            "Momentum (UMD)",
            "Quality (QMJ)",
            "Carry (FX)",
            "Low volatility",
            "Term premium",
            "Credit spread",
        ],
        "Approx. annual\npremium (%)": [
            "5–7", "3–5", "6–9", "4–6",
            "3–5", "3–5", "1–2", "2–4",
        ],
        "Primary mechanism": [
            "Systematic risk / recession exposure",
            "Distress risk / behavioural extrapolation",
            "Slow information diffusion / underreaction",
            "Lottery preference / neglect of quality",
            "Crash risk / global volatility exposure",
            "Leverage aversion / low-beta anomaly",
            "Inflation uncertainty / duration risk",
            "Default & liquidity risk / cyclicality",
        ],
        "Key regime\nvulnerability": [
            "Recessions / valuation compression",
            "Prolonged growth / momentum cycles",
            "Sharp reversals / crowding",
            "Speculative surges (junk rallies)",
            "Risk-off episodes / carry unwinds",
            "Low-vol expensive / sharp recoveries",
            "Inflation shocks / simultaneous sell-off",
            "Credit cycles / liquidity crises",
        ],
    }

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor="white")
    ax.axis("off")

    col_widths = [0.10, 0.12, 0.40, 0.38]
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="left",
        loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.2)

    for j in range(len(df.columns)):
        cell = tbl[0, j]
        cell.set_facecolor("#2166ac")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    for i in range(1, len(df) + 1):
        color = "#f7f7f7" if i % 2 == 0 else "white"
        for j in range(len(df.columns)):
            cell = tbl[i, j]
            cell.set_facecolor(color)
            cell.set_edgecolor("#eeeeee")

    ax.set_title(
        "Major factor premia: mechanism, magnitude, and regime vulnerability\n"
        "Sources: Fama-French, Asness et al., Ilmanen (2011), "
        "Burnside et al.",
        fontsize=9, pad=15, loc="left",
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_factor_table.png",
                bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("  Saved: fig_factor_table.png")


# =========================================================
# Chart 5 — Cumulative factor performance
# =========================================================
def chart_factor_performance() -> None:
    """
    Cumulative performance of Fama-French factors (HML, SMB, UMD)
    versus market, 1963–present.
    """
    ff3_url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Research_Data_Factors_CSV.zip"
    )
    mom_url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Momentum_Factor_CSV.zip"
    )

    print("  Downloading Fama-French factors...")
    try:
        ff3_records = download_ff_zip(ff3_url)
        mom_records = download_ff_zip(mom_url)
    except Exception as e:
        print(f"  Warning: FF download failed: {e}")
        return

    ff3 = pd.DataFrame(
        ff3_records, columns=["date", "MktRF", "SMB", "HML", "RF"]
    ).set_index("date").sort_index()
    mom = pd.DataFrame(
        mom_records, columns=["date", "UMD"]
    ).set_index("date").sort_index()

    for col in ff3.columns:
        ff3[col] = pd.to_numeric(ff3[col], errors="coerce")
    mom["UMD"] = pd.to_numeric(mom["UMD"], errors="coerce")

    start = "1963-07-01"
    ff3   = ff3.loc[start:] / 100
    mom   = mom.loc[start:] / 100

    common = ff3.index.intersection(mom.index)
    ff3    = ff3.loc[common]
    mom    = mom.loc[common]

    mkt_ret = ff3["MktRF"] + ff3["RF"]
    cum_mkt = (1 + mkt_ret).cumprod()
    cum_hml = (1 + ff3["HML"]).cumprod()
    cum_smb = (1 + ff3["SMB"]).cumprod()
    cum_umd = (1 + mom["UMD"]).cumprod()

    freq = 12
    def sharpe(s):
        mu  = float(s.mean()) * freq
        vol = float(s.std(ddof=1)) * np.sqrt(freq)
        return mu / vol if vol > 0 else 0.0

    sh_mkt = sharpe(mkt_ret)
    sh_hml = sharpe(ff3["HML"])
    sh_smb = sharpe(ff3["SMB"])
    sh_umd = sharpe(mom["UMD"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.semilogy(cum_mkt.index, cum_mkt.values,
                color=COLORS["market"], lw=1.8,
                label=f"Market (Sharpe={sh_mkt:.2f})")
    ax.semilogy(cum_hml.index, cum_hml.values,
                color=COLORS["hml"], lw=1.6, linestyle="--",
                label=f"Value HML (Sharpe={sh_hml:.2f})")
    ax.semilogy(cum_smb.index, cum_smb.values,
                color=COLORS["smb"], lw=1.4, linestyle="-.",
                label=f"Size SMB (Sharpe={sh_smb:.2f})")
    ax.semilogy(cum_umd.index, cum_umd.values,
                color=COLORS["umd"], lw=1.4, linestyle=":",
                label=f"Momentum UMD (Sharpe={sh_umd:.2f})")
    ax.axhline(1, color="#cccccc", lw=0.8, linestyle=":")
    ax.set_ylabel("Cumulative return (log scale, start = 1)", labelpad=8)
    ax.set_title(
        "Cumulative factor returns, 1963–present\n"
        "Long-only long-short factors",
        pad=10,
    )
    ax.legend(fontsize=8, framealpha=0.5)

    ax2    = axes[1]
    labels = ["Market", "Value\n(HML)", "Size\n(SMB)", "Momentum\n(UMD)"]
    ann_ret = [
        float(mkt_ret.mean()) * 12 * 100,
        float(ff3["HML"].mean()) * 12 * 100,
        float(ff3["SMB"].mean()) * 12 * 100,
        float(mom["UMD"].mean()) * 12 * 100,
    ]
    ann_vol = [
        float(mkt_ret.std(ddof=1)) * np.sqrt(12) * 100,
        float(ff3["HML"].std(ddof=1)) * np.sqrt(12) * 100,
        float(ff3["SMB"].std(ddof=1)) * np.sqrt(12) * 100,
        float(mom["UMD"].std(ddof=1)) * np.sqrt(12) * 100,
    ]
    colors = [COLORS["market"], COLORS["hml"],
              COLORS["smb"], COLORS["umd"]]
    x     = np.arange(len(labels))
    width = 0.35

    bars1 = ax2.bar(x - width / 2, ann_ret, width,
                    color=colors, alpha=0.85,
                    label="Annualized return (%)",
                    edgecolor="white")
    ax2.bar(x + width / 2, ann_vol, width,
            color=colors, alpha=0.4,
            label="Annualized volatility (%)",
            edgecolor="white", hatch="//")

    for bar, val in zip(bars1, ann_ret):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.2, f"{val:.1f}%",
            ha="center", va="bottom", fontsize=8,
        )

    ax2.axhline(0, color="#999999", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Annualized (%)", labelpad=8)
    ax2.set_title(
        "Factor return and volatility statistics\n"
        "Monthly, 1963–present",
        pad=10,
    )
    ax2.legend(fontsize=8, framealpha=0.5)

    fig.suptitle(
        "Fama-French factor performance, 1963–present\n"
        "Source: Ken French Data Library",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_factor_performance.png",
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_factor_performance.png")


# =========================================================
# Chart 6 — Four-factor decomposition of Magellan Fund
# =========================================================
def chart_factor_decomposition() -> None:
    """
    Run Carhart four-factor regression on Fidelity Magellan (FMAGX)
    monthly returns using public data from Yahoo Finance.
    Show full-sample factor loadings and rolling 36-month alpha.
    """
    ff3_url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Research_Data_Factors_CSV.zip"
    )
    mom_url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Momentum_Factor_CSV.zip"
    )

    print("  Downloading factor data for decomposition...")
    try:
        ff3_records = download_ff_zip(ff3_url)
        mom_records = download_ff_zip(mom_url)
    except Exception as e:
        print(f"  Warning: FF download failed: {e}")
        return

    ff3 = pd.DataFrame(
        ff3_records, columns=["date", "MktRF", "SMB", "HML", "RF"]
    ).set_index("date").sort_index()
    mom = pd.DataFrame(
        mom_records, columns=["date", "UMD"]
    ).set_index("date").sort_index()

    for col in ff3.columns:
        ff3[col] = pd.to_numeric(ff3[col], errors="coerce")
    mom["UMD"] = pd.to_numeric(mom["UMD"], errors="coerce")
    ff3 = ff3 / 100
    mom = mom / 100

    print("  Downloading Fidelity Magellan (FMAGX)...")
    raw = yf.download("FMAGX", start="1980-01-01",
                      progress=False, auto_adjust=True)
    if raw.empty:
        print("  Warning: FMAGX download failed.")
        return

    price    = raw["Close"].squeeze().resample("MS").last().dropna()
    fund_ret = price.pct_change().dropna()
    fund_ret.index = normalize_index(fund_ret.index)

    common = (fund_ret.index
              .intersection(ff3.index)
              .intersection(mom.index))
    fund_ret = fund_ret.loc[common]
    ff3_c    = ff3.loc[common]
    mom_c    = mom.loc[common]

    excess_ret = fund_ret - ff3_c["RF"]

    X = pd.DataFrame({
        "MktRF": ff3_c["MktRF"],
        "SMB":   ff3_c["SMB"],
        "HML":   ff3_c["HML"],
        "UMD":   mom_c["UMD"],
    }).dropna()
    y = excess_ret.reindex(X.index).dropna()
    X = X.reindex(y.index)

    X_const = np.column_stack([np.ones(len(X)), X.values])
    coeffs, _, _, _ = np.linalg.lstsq(X_const, y.values, rcond=None)
    alpha_ann   = coeffs[0] * 12 * 100
    betas       = coeffs[1:]
    beta_labels = ["Market β", "Size\n(SMB)", "Value\n(HML)",
                   "Momentum\n(UMD)"]

    # Rolling 36-month alpha
    window     = 36
    roll_alpha = []
    roll_dates = []
    for i in range(len(y) - window + 1):
        y_w = y.iloc[i:i + window].values
        X_w = X_const[i:i + window]
        c, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
        roll_alpha.append(c[0] * 12 * 100)
        roll_dates.append(y.index[i + window - 1])

    roll_alpha_s = pd.Series(roll_alpha, index=roll_dates)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    colors_bar = [
        COLORS["beta_bar"], COLORS["smb"],
        COLORS["hml"],      COLORS["umd"],
    ]
    bars = ax.bar(beta_labels, betas, color=colors_bar,
                  width=0.5, edgecolor="white")
    for bar, val in zip(bars, betas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01 if val >= 0 else val - 0.03,
            f"{val:.2f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9, fontweight="bold",
        )
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_ylabel("Factor loading (β)", labelpad=8)
    ax.set_title(
        f"Fidelity Magellan (FMAGX) — Carhart factor loadings\n"
        f"Full sample. Annualized alpha = {alpha_ann:.1f}%",
        pad=10,
    )

    ax2 = axes[1]
    ax2.fill_between(
        roll_alpha_s.index, roll_alpha_s.values, 0,
        where=(roll_alpha_s.values > 0),
        alpha=0.25, color=COLORS["factor_bar"],
    )
    ax2.fill_between(
        roll_alpha_s.index, roll_alpha_s.values, 0,
        where=(roll_alpha_s.values <= 0),
        alpha=0.25, color=COLORS["alpha_bar"],
    )
    ax2.plot(roll_alpha_s.index, roll_alpha_s.values,
             color="#333333", lw=1.4,
             label="Rolling 36-month annualized alpha (%)")
    ax2.axhline(0, color="#999999", lw=0.8, linestyle="--")
    ax2.set_ylabel("Rolling annualized alpha (%)", labelpad=8)
    ax2.set_title(
        "Rolling 36-month alpha vs Carhart four-factor model\n"
        "Fidelity Magellan (FMAGX)",
        pad=10,
    )
    ax2.legend(fontsize=8, framealpha=0.5)

    fig.suptitle(
        "Carhart four-factor decomposition: Fidelity Magellan\n"
        "Factor exposures and rolling alpha. "
        "Source: Yahoo Finance, Ken French Data Library.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_factor_decomposition.png",
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_factor_decomposition.png")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    print("Building Chart 1 — Stock-bond correlation (full history)...")
    chart_stock_bond_correlation()

    print("\nBuilding Chart 2 — 60/40 risk contribution...")
    chart_risk_contribution()

    print("\nBuilding Chart 3 — High-yield spread vs forward return...")
    chart_hy_spread_vs_return()

    print("\nBuilding Chart 4 — Factor premia table...")
    chart_factor_table()

    print("\nBuilding Chart 5 — Factor performance vs market...")
    chart_factor_performance()

    print("\nBuilding Chart 6 — Four-factor decomposition Magellan...")
    chart_factor_decomposition()

    print("\nAll charts saved to:", OUT_DIR.resolve())