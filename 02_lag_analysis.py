"""
02_lag_analysis.py — RQ1: how do meteorological drivers lag into lake level,
and does lake elevation modulate that lag?

Notebook:
    import pandas as pd
    daily  = pd.read_csv("output/lakes_daily.csv", parse_dates=["date"])
    static = pd.read_csv("output/lakes_static.csv")
    res = run_lag_analysis(daily, static, sample="N0")

Outputs (into ./output and ./figures):
    ccf_results.csv         full cross-correlation surface, every lake x driver x lag
    peak_lags.csv           peak-correlation lag per lake per driver
    granger_results.csv     Granger causality p-values
    elevation_regression.txt  OLS of peak lag on lake attributes
    duplicate_check.csv     correlation between co-located stations
    figures/*.png

Dependencies: pandas, numpy, scipy, statsmodels, matplotlib.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("output")
FIG = Path("figures")

MAX_LAG = 60          # days of lag to scan
DRIVERS = ["precipitation_mm", "temperature_mean_c", "snowmelt_proxy"]
TARGET = "waterlevel_m"

# 01_build_dataset renames the topographic elevation column, so accept either name.
ELEV_NAMES = ["mean_elevation_m", "gauge_elevation", "elev_mean", "elevation"]


def elevation_series(static):
    """Return gauge_id -> elevation, whatever the column ended up being called."""
    for c in ELEV_NAMES:
        if c in static.columns and static[c].notna().any():
            return pd.to_numeric(static.set_index("gauge_id")[c], errors="coerce"), c
    for c in static.columns:
        if "elev" in c.lower() and static[c].notna().any():
            return pd.to_numeric(static.set_index("gauge_id")[c], errors="coerce"), c
    return pd.Series(dtype=float), None


def rule(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# --------------------------------------------------------- sample construction
def deduplicate(daily, static, sample="N0"):
    """
    One station per water body. Two gauges on the same lake are not independent
    observations - pooling them is pseudo-replication.
    """
    rule(f"SAMPLE CONSTRUCTION (tier filter: {sample})")

    st = static.copy()
    if sample != "ALL":
        st = st[st["regulation_tier"].isin(list(sample) if isinstance(sample, list) else [sample])]

    completeness = "pct_usable_final" if "pct_usable_final" in st.columns else "pct_complete_raw"
    st = st.sort_values(completeness, ascending=False)
    keep = st.drop_duplicates(subset="water_body", keep="first")

    dropped = st[~st["gauge_id"].isin(keep["gauge_id"])]
    if len(dropped):
        print("    dropped as duplicate water bodies:")
        for _, r in dropped.iterrows():
            print(f"      {int(r['gauge_id'])}  {r['water_body']}")

    print(f"\n    {len(st)} stations -> {len(keep)} distinct water bodies")
    d = daily[daily["gauge_id"].isin(keep["gauge_id"])].copy()
    return d, keep.reset_index(drop=True)


def duplicate_check(daily, static):
    """Co-located stations should track each other. If they don't, suspect the datum."""
    rule("DUPLICATE STATION CHECK")
    rows = []
    for wb, grp in static.groupby("water_body"):
        if len(grp) < 2:
            continue
        ids = grp["gauge_id"].tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = daily[daily["gauge_id"] == ids[i]].set_index("date")[TARGET]
                b = daily[daily["gauge_id"] == ids[j]].set_index("date")[TARGET]
                joined = pd.concat([a, b], axis=1, join="inner").dropna()
                if len(joined) < 100:
                    continue
                r = joined.iloc[:, 0].corr(joined.iloc[:, 1])
                dr = joined.iloc[:, 0].diff().corr(joined.iloc[:, 1].diff())
                rows.append({"water_body": wb, "gauge_a": ids[i], "gauge_b": ids[j],
                             "n_days": len(joined), "corr_level": round(r, 4),
                             "corr_daily_change": round(dr, 4),
                             "mean_offset_m": round(
                                 (joined.iloc[:, 0] - joined.iloc[:, 1]).mean(), 3)})
    if not rows:
        print("    no co-located stations in this sample")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\n    corr_level near 1 = same signal. A large mean_offset_m with high")
    print("    correlation means different gauge datums, not different behaviour.")
    df.to_csv(OUT / "duplicate_check.csv", index=False)
    return df


# ------------------------------------------------------------ seasonal anomaly
def deseasonalise(s, dates):
    """
    Remove the day-of-year climatology.

    This matters: raw lake level and raw precipitation both carry a strong annual
    cycle, so their cross-correlation mostly measures 'both are seasonal' rather
    than any causal lag. Working on anomalies is what makes the CCF interpretable.
    """
    doy = dates.dt.dayofyear
    clim = s.groupby(doy).transform("mean")
    return s - clim


# ------------------------------------------------------------------------ CCF
def cross_correlation(target, driver, max_lag=MAX_LAG):
    """Correlate driver at t-k against target at t, for k = 0..max_lag."""
    out = []
    for k in range(max_lag + 1):
        d = driver.shift(k)
        pair = pd.concat([target, d], axis=1).dropna()
        if len(pair) < 200:
            out.append(np.nan)
            continue
        out.append(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
    return np.array(out)


def run_ccf(daily, static):
    rule("CROSS-CORRELATION: DRIVERS vs LEVEL")
    ccf_rows, peak_rows = [], []

    for _, meta in static.iterrows():
        gid = int(meta["gauge_id"])
        g = daily[daily["gauge_id"] == gid].sort_values("date").copy()
        if len(g) < 1000:
            continue

        lvl_a = deseasonalise(g[TARGET], g["date"])
        for drv in DRIVERS:
            if drv not in g.columns or g[drv].notna().sum() < 1000:
                continue
            drv_a = deseasonalise(g[drv], g["date"])
            ccf = cross_correlation(lvl_a.reset_index(drop=True),
                                    drv_a.reset_index(drop=True))
            if np.all(np.isnan(ccf)):
                continue

            for k, v in enumerate(ccf):
                ccf_rows.append({"gauge_id": gid, "water_body": meta["water_body"],
                                 "driver": drv, "lag_days": k, "corr": v})

            peak_k = int(np.nanargmax(np.abs(ccf)))
            peak_rows.append({
                "gauge_id": gid,
                "water_body": meta["water_body"],
                "regulation_tier": meta.get("regulation_tier"),
                "driver": drv,
                "peak_lag_days": peak_k,
                "peak_corr": round(float(ccf[peak_k]), 4),
                "corr_at_lag0": round(float(ccf[0]), 4),
                "peak_sign": "positive" if ccf[peak_k] > 0 else "negative",
                # a peak within 5 days of the scan limit may be truncated:
                # the true maximum could lie beyond MAX_LAG
                "peak_at_boundary": peak_k >= MAX_LAG - 5,
            })

    ccf_df = pd.DataFrame(ccf_rows)
    peak_df = pd.DataFrame(peak_rows)
    ccf_df.to_csv(OUT / "ccf_results.csv", index=False)
    peak_df.to_csv(OUT / "peak_lags.csv", index=False)

    for drv in DRIVERS:
        sub = peak_df[peak_df["driver"] == drv]
        if len(sub):
            print(f"\n    {drv}")
            print(sub[["water_body", "peak_lag_days", "peak_corr", "peak_sign",
                       "peak_at_boundary"]]
                  .sort_values("peak_lag_days").to_string(index=False))
            edge = sub[sub["peak_at_boundary"]]
            if len(edge):
                print(f"    !! peak at scan limit for: {list(edge['water_body'])}")
                print(f"       True peak may exceed {MAX_LAG} days - say so, or raise MAX_LAG.")
    return ccf_df, peak_df


# -------------------------------------------------------------------- Granger
def run_granger(daily, static, maxlag=30):
    rule("GRANGER CAUSALITY (on stationary first differences)")
    try:
        from statsmodels.tsa.stattools import grangercausalitytests, adfuller
    except ImportError:
        print("    statsmodels not installed - skipping")
        return pd.DataFrame()

    rows = []
    for _, meta in static.iterrows():
        gid = int(meta["gauge_id"])
        g = daily[daily["gauge_id"] == gid].sort_values("date")
        for drv in DRIVERS:
            if drv not in g.columns:
                continue
            pair = pd.DataFrame({
                "y": g[TARGET].diff(),
                "x": g[drv].diff(),
            }).dropna()
            if len(pair) < 500 or pair["x"].std() == 0:
                continue
            try:
                adf_p = adfuller(pair["y"], autolag="AIC")[1]
                res = grangercausalitytests(pair[["y", "x"]], maxlag=maxlag)
                pvals = {k: v[0]["ssr_ftest"][1] for k, v in res.items()}
                best = min(pvals, key=pvals.get)
                rows.append({"gauge_id": gid, "water_body": meta["water_body"],
                             "driver": drv, "adf_p_target": round(adf_p, 5),
                             "best_lag": best, "min_p": pvals[best],
                             "significant_5pct": pvals[best] < 0.05})
            except Exception as e:
                print(f"    {gid} {drv}: {e}")

    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(OUT / "granger_results.csv", index=False)
        print(df.to_string(index=False))
        print("\n    Note: Granger causality is predictive precedence, not physical")
        print("    causation. Phrase it that way in the write-up.")
    return df


# --------------------------------------------------- does elevation drive lag?
def elevation_regression(peak_df, static):
    rule("PEAK LAG vs LAKE ATTRIBUTES")
    try:
        import statsmodels.api as sm
    except ImportError:
        print("    statsmodels not installed - skipping")
        return

    elev, elev_col = elevation_series(static)
    if elev_col:
        print(f"    elevation column resolved to: {elev_col}")
    else:
        print("    !! no elevation column found - the headline test cannot run")

    candidates = ELEV_NAMES + ["area", "catchment_area_km2", "frac_snow",
                               "glac_area", "p_mean", "aridity", "mean_slope"]
    have = [c for c in dict.fromkeys(candidates) if c in static.columns
            and static[c].notna().any()]
    print(f"    predictors available: {have}")

    lines = []
    for drv in DRIVERS:
        sub = peak_df[peak_df["driver"] == drv].merge(
            static[["gauge_id"] + have], on="gauge_id", how="left")
        sub = sub.dropna(subset=["peak_lag_days"])
        if len(sub) < 6:
            continue

        # Bivariate first - at n=14 this is the defensible result.
        header = f"\n### {drv}: peak lag vs each attribute   (n={len(sub)})"
        print(header)
        lines.append(header)
        for c in have:
            x = pd.to_numeric(sub[c], errors="coerce")
            if x.notna().sum() < 6 or x.std(skipna=True) == 0:
                continue
            r = sub["peak_lag_days"].astype(float).corr(x)
            rs = sub["peak_lag_days"].astype(float).corr(x, method="spearman")
            line = f"    r(peak_lag, {c:<20}) = {r:+.3f}   spearman = {rs:+.3f}"
            print(line)
            lines.append(line)

        # Sign of the peak correlation against elevation - the mechanism test.
        if elev_col and "peak_sign" in sub.columns:
            sub["_elev"] = sub["gauge_id"].map(elev)
            pos = sub[sub["peak_sign"] == "positive"]["_elev"].dropna()
            neg = sub[sub["peak_sign"] == "negative"]["_elev"].dropna()
            if len(pos) and len(neg):
                msg = (f"    peak correlation POSITIVE at mean elevation {pos.mean():.0f} m "
                       f"(n={len(pos)}), NEGATIVE at {neg.mean():.0f} m (n={len(neg)})")
                print(msg)
                lines.append(msg)

        preds = [c for c in have if pd.to_numeric(sub[c], errors="coerce").notna().sum()
                 >= len(sub) - 1 and pd.to_numeric(sub[c], errors="coerce").std() > 0]
        if elev_col and elev_col in preds:                 # keep elevation, drop others
            preds = [elev_col] + [p for p in preds if p != elev_col]
        preds = preds[:3]
        if not preds:
            continue

        Xv = sub[preds].apply(pd.to_numeric, errors="coerce")
        X = sm.add_constant(Xv.fillna(Xv.mean()))
        y = sub["peak_lag_days"].astype(float)
        model = sm.OLS(y, X).fit()
        txt = f"\n### OLS: peak lag of {drv} ~ {' + '.join(preds)}   (n={len(sub)})"
        print(txt)
        print(model.summary().as_text())
        lines.append(txt + "\n" + model.summary().as_text())

    (OUT / "elevation_regression.txt").write_text("\n\n".join(lines), encoding="utf-8")
    print(f"\n    -> {OUT / 'elevation_regression.txt'}")
    print("    With n around 14, treat multivariate coefficients as descriptive.")
    print("    The bivariate correlation is the defensible headline.")


# ------------------------------------------------------------------- figures
def make_figures(ccf_df, peak_df, static, daily):
    rule("FIGURES")
    FIG.mkdir(exist_ok=True)

    elev, elev_col = elevation_series(static)
    if len(elev.dropna()):
        lo, hi = float(np.nanmin(elev)), float(np.nanmax(elev))
    else:
        lo = hi = np.nan

    def shade_for(gid):
        e = elev.get(gid, np.nan)
        if not np.isfinite(e) or not np.isfinite(lo) or hi == lo:
            return 0.5, e
        return (e - lo) / (hi - lo), e

    # Fig 1: CCF curves, coloured by elevation
    for drv in DRIVERS:
        sub = ccf_df[ccf_df["driver"] == drv]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for gid in sub["gauge_id"].unique():
            s = sub[sub["gauge_id"] == gid]
            shade, e = shade_for(gid)
            ax.plot(s["lag_days"], s["corr"], color=plt.cm.viridis(shade),
                    lw=1.4, alpha=0.85,
                    label=f"{s['water_body'].iloc[0]} ({e:.0f} m)"
                    if np.isfinite(e) else s["water_body"].iloc[0])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Correlation (seasonal anomalies)")
        ax.set_title(f"Cross-correlation: {drv} leading lake level")
        ax.legend(fontsize=6.5, ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / f"ccf_{drv}.png", dpi=180)
        plt.close(fig)
        print(f"    figures/ccf_{drv}.png")

    # Fig 2: peak lag against elevation
    if len(elev):
        fig, axes = plt.subplots(1, len(DRIVERS), figsize=(4.2 * len(DRIVERS), 4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, drv in zip(axes, DRIVERS):
            sub = peak_df[peak_df["driver"] == drv]
            if sub.empty:
                continue
            x = [elev.get(g, np.nan) for g in sub["gauge_id"]]
            ax.scatter(x, sub["peak_lag_days"], s=45, alpha=0.8, edgecolor="k", lw=0.5)
            for xi, yi, nm in zip(x, sub["peak_lag_days"], sub["water_body"]):
                if np.isfinite(xi):
                    ax.annotate(nm[:11], (xi, yi), fontsize=5.5,
                                xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel("Lake elevation (m)")
            ax.set_title(drv, fontsize=9)
        axes[0].set_ylabel("Peak-correlation lag (days)")
        fig.tight_layout()
        fig.savefig(FIG / "peak_lag_vs_elevation.png", dpi=180)
        plt.close(fig)
        print("    figures/peak_lag_vs_elevation.png")

    # Fig 3: annual cycle per lake, standardised so lakes are comparable
    fig, ax = plt.subplots(figsize=(9, 5))
    for gid, g in daily.groupby("gauge_id"):
        s = g.set_index("date")[TARGET]
        z = (s - s.mean()) / s.std()
        clim = z.groupby(z.index.dayofyear).mean()
        shade, _ = shade_for(gid)
        ax.plot(clim.index, clim.values, lw=1.3, alpha=0.85, color=plt.cm.viridis(shade))
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Standardised level anomaly")
    ax.set_title("Annual cycle by lake (dark = low elevation, bright = high)")
    fig.tight_layout()
    fig.savefig(FIG / "annual_cycle.png", dpi=180)
    plt.close(fig)
    print("    figures/annual_cycle.png")


# -------------------------------------------------------------------- driver
def run_lag_analysis(daily, static, sample="N0", tag=None):
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    tag = tag or (sample if isinstance(sample, str) else "custom")

    duplicate_check(daily, static)
    d, st = deduplicate(daily, static, sample=sample)
    ccf_df, peak_df = run_ccf(d, st)
    granger_df = run_granger(d, st)
    elevation_regression(peak_df, st)
    make_figures(ccf_df, peak_df, st, d)

    # keep each run's outputs separate
    st.to_csv(OUT / f"sample_{tag}.csv", index=False)
    for name in ("ccf_results", "peak_lags", "granger_results",
                 "elevation_regression", "duplicate_check"):
        for ext in (".csv", ".txt"):
            src = OUT / f"{name}{ext}"
            if src.exists():
                src.replace(OUT / f"{name}_{tag}{ext}")
    for p in FIG.glob("*.png"):
        if f"_{tag}" not in p.stem:
            p.replace(FIG / f"{p.stem}_{tag}.png")

    rule(f"DONE - RQ1 EVIDENCE COMPLETE (tag: {tag})")
    print(f"    outputs suffixed _{tag}")
    return {"daily": d, "static": st, "ccf": ccf_df,
            "peaks": peak_df, "granger": granger_df, "tag": tag}
