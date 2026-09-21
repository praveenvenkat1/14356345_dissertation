"""
05_figures.py
generates Figures 1, 6, 7 and 8
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("output")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)

DPI = 300
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODEL_ORDER = ["persistence", "climatology", "ridge", "lightgbm", "lstm"]
MODEL_LABEL = {
    "persistence": "Persistence", "climatology": "Climatology",
    "ridge": "Ridge", "lightgbm": "LightGBM", "lstm": "LSTM",
}
MODEL_COLOUR = {
    "persistence": "#9e9e9e", "climatology": "#c9a227",
    "ridge": "#4c72b0", "lightgbm": "#55a868", "lstm": "#c44e52",
}

ELEV_NAMES = ["mean_elevation_m", "gauge_elevation", "elev_mean", "elevation"]


def _elev_col(df):
    for c in ELEV_NAMES:
        if c in df.columns and df[c].notna().any():
            return c
    for c in df.columns:
        if "elev" in c.lower() and df[c].notna().any():
            return c
    return None


def _sample(tag="N0"):
    p = OUT / f"sample_{tag}.csv"
    if not p.exists():
        p = OUT / "lakes_static.csv"
    return pd.read_csv(p)


def _save(fig, name):
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")



# FIGURE 1 — Study area
def fig1_study_area(camels_root, tag="N0"):
    """The selected lakes on a Swiss outline, sized by catchment area,
    coloured by elevation. Goes in Section 3.5."""
    print("Figure 1: study area")
    st = _sample(tag)
    ec = _elev_col(st)
    root = Path(camels_root)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    plotted_background = False

    try:
        import geopandas as gpd

        # Catchment polygons give the Swiss outline for free
        cat_files = list(root.rglob("*CAMELS_CH_catchments.shp"))
        if cat_files:
            cat = gpd.read_file(cat_files[0]).to_crs(4326)
            cat.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.3)
            sel = cat[cat["gauge_id"].astype(float).isin(st["gauge_id"])] \
                if "gauge_id" in cat.columns else None
            if sel is not None and len(sel):
                sel.plot(ax=ax, color="#dce7ec", edgecolor="#7aa0b2", linewidth=0.7)
            plotted_background = True

        stn_files = list(root.rglob("*gauging_stations.shp"))
        if stn_files:
            stn = gpd.read_file(stn_files[0]).to_crs(4326)
            stn["gauge_id"] = pd.to_numeric(stn["gauge_id"], errors="coerce")
            stn = stn[stn["gauge_id"].isin(st["gauge_id"])]
            st = st.merge(
                pd.DataFrame({"gauge_id": stn["gauge_id"],
                              "lon": stn.geometry.x, "lat": stn.geometry.y}),
                on="gauge_id", how="left")
    except ImportError:
        print("    geopandas absent — plotting points only")

    if "lon" not in st.columns or st["lon"].isna().all():
        print("    !! no coordinates available; Figure 1 cannot be drawn")
        plt.close(fig)
        return

    area = st["catchment_area_km2"] if "catchment_area_km2" in st.columns else None
    sizes = (40 + 260 * (area / area.max()) ** 0.5) if area is not None else 90
    elev = pd.to_numeric(st[ec], errors="coerce") if ec else None

    sc = ax.scatter(st["lon"], st["lat"], s=sizes,
                    c=elev if elev is not None else "#c44e52",
                    cmap="viridis", edgecolor="black", linewidth=0.6,
                    zorder=5, alpha=0.92)

    for _, r in st.iterrows():
        if pd.notna(r.get("lon")):
            ax.annotate(str(r["water_body"]).replace("_", " "),
                        (r["lon"], r["lat"]), fontsize=6.5, zorder=6,
                        xytext=(5, 4), textcoords="offset points")

    if elev is not None:
        cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label("Lake elevation (m a.s.l.)", fontsize=8)

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Study area: {len(st)} catchment-unregulated Swiss lakes\n"
                 "Marker size proportional to catchment area", loc="left")
    ax.set_aspect(1 / np.cos(np.deg2rad(float(st["lat"].mean()))))
    if not plotted_background:
        ax.grid(alpha=0.25, linestyle=":")
    _save(fig, "fig1_study_area")



# FIGURE 6 — Observed versus predicted
def fig6_observed_vs_predicted(camels_root, horizon=7, tag="N0",
                               alpine="Silsersee", lowland="Hallwilersee",
                               year=2018):
    """Refits the LSTM at one horizon and plots one alpine against one
    lowland lake over a single year. Section 4.5.

    Takes a few minutes — it retrains. Seeds are fixed so the model matches
    the one reported.
    """
    print(f"Figure 6: observed vs predicted (h={horizon})")
    import importlib
    m = importlib.import_module("04_models") if Path("04_models.py").exists() else None
    if m is None:
        print("    !! 04_models.py not found in the working directory")
        return

    static = pd.read_csv(OUT / f"sample_{tag}.csv")
    daily = pd.read_csv(OUT / "lakes_daily.csv", parse_dates=["date"])
    daily = daily[daily["gauge_id"].isin(static["gauge_id"])]

    lakes, dyn_cols, stat_cols = m.prepare(daily, static)
    sets = m.windows(lakes, horizon)
    preds, meta = m.lstm(lakes, sets, horizon, len(stat_cols))
    if preds is None:
        print("    !! LSTM unavailable")
        return

    by_lake = {}
    for (g, t), pv in zip(meta, preds):
        by_lake.setdefault(g, ([], []))
        by_lake[g][0].append(t)
        by_lake[g][1].append(pv)

    def series_for(name):
        match = static[static["water_body"].astype(str)
                       .str.lower().str.contains(name.lower().replace(" ", "_"))]
        if match.empty:
            return None
        g = int(match.iloc[0]["gauge_id"])
        if g not in by_lake:
            return None
        L = lakes[g]
        ts = np.array(by_lake[g][0])
        yhat = np.array(by_lake[g][1]) * L["sd"] + L["mu"]
        dates = pd.to_datetime(L["dates"][ts + horizon])
        df = pd.DataFrame({
            "date": dates,
            "observed": L["y_raw"][ts + horizon],
            "predicted": yhat,
            "persistence": L["y_raw"][ts],
        }).sort_values("date")
        return L["name"], df

    panels = [p for p in (series_for(alpine), series_for(lowland)) if p]
    if not panels:
        print("    !! neither named lake found in the sample")
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(7.5, 2.6 * len(panels)),
                             sharex=False)
    axes = np.atleast_1d(axes)
    for ax, (name, df) in zip(axes, panels):
        d = df[df["date"].dt.year == year]
        if d.empty:
            d = df
        ax.plot(d["date"], d["observed"], color="black", lw=1.3, label="Observed")
        ax.plot(d["date"], d["predicted"], color=MODEL_COLOUR["lstm"],
                lw=1.1, label="LSTM forecast")
        ax.plot(d["date"], d["persistence"], color="#9e9e9e", lw=0.9,
                linestyle="--", label="Persistence")
        ax.set_title(str(name).replace("_", " "), loc="left")
        ax.set_ylabel("Level (m)")
        ax.margins(x=0.01)
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    axes[-1].set_xlabel(f"Date ({year})")
    fig.suptitle(f"Observed and {horizon}-day forecast water level",
                 x=0.005, ha="left", fontsize=10.5)
    fig.tight_layout()
    _save(fig, f"fig6_observed_vs_predicted_h{horizon}")



# FIGURE 7 — Persistence skill score by model and horizon
def fig7_skill_by_horizon(tag="N0", drop_climatology=True):
    """Grouped bars of median PSS with per-lake points overlaid, so the
    spread behind the median is visible. Section 4.5.1."""
    print("Figure 7: skill by model and horizon")
    r = pd.read_csv(OUT / f"model_results_{tag}.csv")
    r = r[r["model"] != "persistence"]
    if drop_climatology:
        # climatology sits near -6 and would flatten everything else
        r = r[r["model"] != "climatology"]

    models = [m for m in MODEL_ORDER if m in r["model"].unique()]
    horizons = sorted(r["horizon"].unique())
    x = np.arange(len(horizons))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i, mdl in enumerate(models):
        offs = x + (i - (len(models) - 1) / 2) * width
        med = [r[(r.model == mdl) & (r.horizon == h)]["pss"].median() for h in horizons]
        ax.bar(offs, med, width * 0.9, label=MODEL_LABEL[mdl],
               color=MODEL_COLOUR[mdl], edgecolor="black", linewidth=0.5, zorder=3)
        for xi, h in zip(offs, horizons):
            pts = r[(r.model == mdl) & (r.horizon == h)]["pss"].dropna()
            jitter = np.random.RandomState(0).normal(0, width * 0.10, len(pts))
            ax.scatter(np.full(len(pts), xi) + jitter, pts, s=7,
                       color="black", alpha=0.35, zorder=4, linewidths=0)

    ax.axhline(0, color="black", lw=1.0, zorder=5)
    ax.text(0.005, 0.015, "Persistence baseline", transform=ax.get_yaxis_transform(),
            fontsize=7.5, color="#444444", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} day" if h == 1 else f"{h} days" for h in horizons])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Persistence skill score")
    ax.set_title("Forecast skill relative to persistence\n"
                 "Bars show the median across lakes; points show individual lakes",
                 loc="left")
    ax.legend(frameon=False, ncol=len(models))
    ax.grid(axis="y", alpha=0.25, linestyle=":", zorder=0)
    _save(fig, "fig7_skill_by_horizon")



# FIGURE 8 — Skill by season and extreme
def fig8_skill_heatmap(tag="N0"):
    """Diverging heatmap of median PSS by stratum. Red cells mark the
    conditions where models lose to persistence. Section 4.6."""
    print("Figure 8: skill by season and extreme")
    s = pd.read_csv(OUT / f"model_strata_{tag}.csv")

    order = ["high_5pct", "season_winter", "season_snowmelt",
             "season_summer", "season_autumn", "low_5pct"]
    labels = {
        "high_5pct": "High water (top 5%)", "low_5pct": "Low water (bottom 5%)",
        "season_winter": "Winter", "season_snowmelt": "Snowmelt (Mar–May)",
        "season_summer": "Summer", "season_autumn": "Autumn",
    }
    models = [m for m in MODEL_ORDER if m in s["model"].unique()]
    horizons = sorted(s["horizon"].unique())

    piv = s.pivot_table(index=["model", "stratum"], columns="horizon",
                        values="pss", aggfunc="median")

    rows, ylabels, seps = [], [], []
    for mi, mdl in enumerate(models):
        for st_key in order:
            if (mdl, st_key) in piv.index:
                rows.append([piv.loc[(mdl, st_key), h] for h in horizons])
                ylabels.append(labels.get(st_key, st_key))
        if mi < len(models) - 1:
            seps.append(len(rows) - 0.5)
    M = np.array(rows, dtype=float)

    lim = float(np.nanmax(np.abs(M)))
    fig, ax = plt.subplots(figsize=(6.2, 0.34 * len(rows) + 2.0))
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                shade = "white" if abs(M[i, j]) > 0.62 * lim else "black"
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.5, color=shade)

    for y in seps:
        ax.axhline(y, color="black", lw=1.4)

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"{h} d" for h in horizons])
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Forecast horizon")

    # model group labels down the left edge
    n_per = len(rows) // max(len(models), 1)
    for mi, mdl in enumerate(models):
        ax.text(-0.62, mi * n_per + (n_per - 1) / 2, MODEL_LABEL[mdl],
                rotation=90, va="center", ha="center",
                fontsize=9, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cb.set_label("Median persistence skill score", fontsize=8)
    ax.set_title("Forecast skill by hydrological regime\n"
                 "Blue beats persistence; red is worse than assuming no change",
                 loc="left")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    _save(fig, "fig8_skill_heatmap")


# =====================================================================
def make_all(camels_root, tag="N0", skip_fig6=False):
    fig1_study_area(camels_root, tag)
    fig7_skill_by_horizon(tag)
    fig8_skill_heatmap(tag)
    if not skip_fig6:
        fig6_observed_vs_predicted(camels_root, horizon=7, tag=tag)
    print("\nDone. Check figures/ before pasting into the report.")
