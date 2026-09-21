"""
01_build_dataset.py 
builds the curated natural-lake dataset from CAMELS-CH
"""

import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- parameters
# Documented, reproducible selection criteria. Change them here, not inline,
# so your methodology section matches your code.
MIN_COMPLETENESS = 90.0    # % non-missing water level required
FLATLINE_RUN     = 5       # identical consecutive values = stuck sensor
MAD_WINDOW       = 30      # days, rolling window for spike detection
MAD_THRESHOLD    = 5.0     # modified z-score above which a point is a spike
GAP_SHORT        = 3       # <= this many days: linear interpolation
GAP_MEDIUM       = 14      # <= this many days: seasonal-naive + interpolation
                           # > GAP_MEDIUM: left as NaN and flagged
SPECIFIC_STORAGE_N1 = 10.0 # mm; upstream reservoir capacity per unit catchment

OUT = Path("output")


# ------------------------------------------------------------------ plumbing
def rule(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def read_semicolon(path, skip_comment=True):
    
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, encoding=enc) as fh:
                first = fh.readline()
            skip = 1 if (skip_comment and first.lstrip().startswith("#")) else 0
            df = pd.read_csv(path, sep=";", skiprows=skip, encoding=enc)
            if df.shape[1] > 1:
                return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise IOError(f"could not parse {path.name}")


def find_col(df, *keywords, required=False, label=""):
    
    def norm(s):
        s = unicodedata.normalize("NFKD", str(s))
        return "".join(c for c in s if not unicodedata.combining(c)).lower()

    for kw in keywords:
        for c in df.columns:
            if norm(kw) in norm(c):
                return c
    if required:
        raise KeyError(f"no column matching {keywords} for {label}. Columns: {list(df.columns)}")
    return None


def load_stations(root):
    hits = list(root.rglob("*gauging_stations*.dbf"))
    if not hits:
        raise FileNotFoundError("CAMELS_CH_gauging_stations.dbf not found")
    path = hits[0]
    try:
        import geopandas as gpd
        gdf = gpd.read_file(path)
        df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        # keep coordinates if present
        try:
            pts = gdf.geometry.to_crs(4326)
            df["lon"], df["lat"] = pts.x.values, pts.y.values
        except Exception:
            pass
    except ImportError:
        import shapefile
        r = shapefile.Reader(str(path.with_suffix("")))
        df = pd.DataFrame(r.records(), columns=[f[0] for f in r.fields[1:]])
    df["gauge_id"] = pd.to_numeric(df["gauge_id"], errors="coerce").astype("Int64")
    return df


# ----------------------------------------------------------- 1. lake selection
def select_lakes(root):
    rule("1. SELECTING LAKES")
    st = load_stations(root)
    lakes = st[(st["type"].str.lower() == "lake") & (st["country"] == "CH")].copy()
    print(f"    {len(st)} stations -> {len(lakes)} Swiss lake stations")
    return lakes.reset_index(drop=True)


# ------------------------------------------------------- 2. regulation tiering
def classify_regulation(root, lakes):
    """
      N0  no upstream hydropower and no upstream reservoir
      N1  some upstream storage, but specific capacity < SPECIFIC_STORAGE_N1 mm
      R   substantial upstream anthropogenic storage
    """
    rule("2. REGULATION TIERING")
    hi = read_semicolon(next(root.rglob("*humaninfluence*.csv")))
    topo = read_semicolon(next(root.rglob("*topographic_attributes.csv")))

    hi["gauge_id"] = pd.to_numeric(hi["gauge_id"], errors="coerce").astype("Int64")
    topo["gauge_id"] = pd.to_numeric(topo["gauge_id"], errors="coerce").astype("Int64")

    area_col = find_col(topo, "area", required=True, label="catchment area")
    elev_col = find_col(topo, "elev")
    slope_col = find_col(topo, "slope")
    print(f"    area column:  {area_col}")
    print(f"    elev column:  {elev_col}")

    keep_topo = ["gauge_id", area_col] + [c for c in (elev_col, slope_col) if c]
    df = lakes.merge(hi, on="gauge_id", how="left").merge(
        topo[keep_topo], on="gauge_id", how="left")

    df = df.rename(columns={area_col: "catchment_area_km2"})
    if elev_col:
        df = df.rename(columns={elev_col: "mean_elevation_m"})
    if slope_col:
        df = df.rename(columns={slope_col: "mean_slope"})

    for c in ("hp_count", "num_reservoir", "reservoir_cap"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)

    # reservoir capacity (m3) spread over the catchment, expressed in mm
    area_m2 = pd.to_numeric(df["catchment_area_km2"], errors="coerce") * 1e6
    df["specific_storage_mm"] = (df["reservoir_cap"] / area_m2 * 1000).replace(
        [np.inf, -np.inf], np.nan).round(3)

    def tier(r):
        if r["hp_count"] == 0 and r["num_reservoir"] == 0:
            return "N0"
        if pd.notna(r["specific_storage_mm"]) and r["specific_storage_mm"] < SPECIFIC_STORAGE_N1:
            return "N1"
        return "R"

    df["regulation_tier"] = df.apply(tier, axis=1)
    print("\n" + df["regulation_tier"].value_counts().to_string())
    print("\n", df[["gauge_id", "water_body", "regulation_tier",
                    "hp_count", "num_reservoir", "specific_storage_mm"]]
          .sort_values("regulation_tier").to_string(index=False))
    return df


# ---------------------------------------------------- 3. attach more attributes
def attach_attributes(root, static):
    rule("3. ATTACHING STATIC ATTRIBUTES")
    wanted = {
        "glacier": ["glac_area", "glac_perc", "glacier"],
        "climate": ["p_mean", "t_mean", "frac_snow", "aridity"],
        "landcover": ["ice_perc", "rock_perc", "inwater_perc", "crop_perc"],
    }
    for key, cols in wanted.items():
        hits = [p for p in (root / "static_attributes").glob("*.csv") if key in p.name.lower()]
        if not hits:
            continue
        try:
            df = read_semicolon(hits[0])
        except IOError:
            print(f"    !! skipped {hits[0].name}")
            continue
        df["gauge_id"] = pd.to_numeric(df["gauge_id"], errors="coerce").astype("Int64")
        found = [c for c in df.columns
                 if c != "gauge_id" and any(k in c.lower() for k in cols)]
        if found:
            static = static.merge(df[["gauge_id"] + found], on="gauge_id", how="left")
            print(f"    {hits[0].name}: +{len(found)} cols -> {found}")
    return static


# ------------------------------------------------------------------- 4. QC
def qc_series(s):
    
    flag = pd.Series("ok", index=s.index, dtype=object)
    flag[s.isna()] = "missing"

    # stuck sensor: runs of identical values
    same = s.eq(s.shift()) & s.notna()
    grp = (~same).cumsum()
    runlen = same.groupby(grp).transform("sum") + 1
    flag[(runlen >= FLATLINE_RUN) & s.notna()] = "flatline"

    # spikes via rolling median absolute deviation (robust to skew, unlike z-scores)
    med = s.rolling(MAD_WINDOW, center=True, min_periods=5).median()
    mad = (s - med).abs().rolling(MAD_WINDOW, center=True, min_periods=5).median()
    mz = 0.6745 * (s - med) / mad.replace(0, np.nan)
    flag[(mz.abs() > MAD_THRESHOLD) & s.notna()] = "spike"

    clean = s.copy()
    clean[flag.isin(["spike", "flatline"])] = np.nan

    # gap taxonomy
    isna = clean.isna()
    gid = (~isna).cumsum()
    gaplen = isna.groupby(gid).transform("sum")

    short = isna & (gaplen <= GAP_SHORT)
    medium = isna & (gaplen > GAP_SHORT) & (gaplen <= GAP_MEDIUM)
    long_ = isna & (gaplen > GAP_MEDIUM)

    filled = clean.interpolate(limit=GAP_SHORT, limit_area="inside")
    if medium.any():
        doy_clim = clean.groupby(clean.index.dayofyear).transform("mean")
        filled[medium] = doy_clim[medium]
        filled = filled.interpolate(limit=GAP_MEDIUM, limit_area="inside")
    filled[long_] = np.nan          # never fabricate the target across long gaps

    flag[short & filled.notna()] = "interp_short"
    flag[medium & filled.notna()] = "interp_seasonal"
    flag[long_] = "gap_long"
    return filled, flag


# -------------------------------------------------------- 5. feature building
def build_features(df):
    
    d = df.sort_values("date").copy()
    p, t, swe, lvl = "precipitation_mm", "temperature_mean_c", "swe_mm", "waterlevel_m"

    # Antecedent Precipitation Index: exponentially weighted recent rainfall
    for hl in (30, 60, 90):
        d[f"api_{hl}"] = d[p].ewm(halflife=hl, min_periods=1).mean()

    # Snowmelt proxy: SWE lost on days warm enough to melt
    swe_drop = (d[swe].shift(1) - d[swe]).clip(lower=0)
    d["snowmelt_proxy"] = swe_drop.where(d[t] > 0, 0.0)
    d["snowmelt_cum30"] = d["snowmelt_proxy"].rolling(30, min_periods=1).sum()

    # Thermal forcing
    d["degree_days"] = d[t].clip(lower=0)
    d["cdd_30"] = d["degree_days"].rolling(30, min_periods=1).sum()

    # Autoregressive structure
    for lag in (1, 3, 7, 14, 30):
        d[f"level_lag{lag}"] = d[lvl].shift(lag)
    for w in (7, 30):
        d[f"level_roll{w}_mean"] = d[lvl].shift(1).rolling(w, min_periods=1).mean()
        d[f"level_roll{w}_std"] = d[lvl].shift(1).rolling(w, min_periods=1).std()
    d["level_delta1"] = d[lvl].diff()

    # Seasonality without a January discontinuity
    doy = d["date"].dt.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return d


# ------------------------------------------------------------ 6. daily panel
def build_panel(root, static):
    rule("4-5. DAILY PANEL, QC AND FEATURES")
    ts_dir = next(d for d in root.rglob("observation_based") if d.is_dir())
    frames, qc_rows = [], []

    for _, row in static.iterrows():
        gid = int(row["gauge_id"])
        f = ts_dir / f"CAMELS_CH_obs_based_{gid}.csv"
        if not f.exists():
            print(f"    !! missing {f.name}")
            continue

        raw = pd.read_csv(f, sep=";", encoding="latin-1")
        raw.columns = [c.strip() for c in raw.columns]
        ren = {
            find_col(raw, "date", required=True, label=gid): "date",
            find_col(raw, "waterlevel", required=True, label=gid): "waterlevel_m",
            find_col(raw, "precipitation"): "precipitation_mm",
            find_col(raw, "temperature_min"): "temperature_min_c",
            find_col(raw, "temperature_mean"): "temperature_mean_c",
            find_col(raw, "temperature_max"): "temperature_max_c",
            find_col(raw, "swe"): "swe_mm",
            find_col(raw, "rel_sun"): "rel_sun_dur_pct",
            find_col(raw, "discharge_vol"): "discharge_m3s",
        }
        raw = raw.rename(columns={k: v for k, v in ren.items() if k})
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.dropna(subset=["date"]).set_index("date").sort_index()

        for c in ("waterlevel_m", "precipitation_mm", "temperature_mean_c", "swe_mm"):
            if c not in raw:
                raw[c] = np.nan
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

        pct = 100 * raw["waterlevel_m"].notna().mean()
        if pct < MIN_COMPLETENESS:
            print(f"    {gid} {row['water_body'][:24]:<24} dropped ({pct:.1f}% complete)")
            continue

        cleaned, flags = qc_series(raw["waterlevel_m"])
        raw["waterlevel_raw_m"] = raw["waterlevel_m"]
        raw["waterlevel_m"] = cleaned
        raw["qc_flag"] = flags.values

        out = raw.reset_index()
        out.insert(0, "gauge_id", gid)
        out.insert(1, "water_body", row["water_body"])
        out.insert(2, "regulation_tier", row["regulation_tier"])
        out = build_features(out)
        frames.append(out)

        counts = flags.value_counts()
        qc_rows.append({
            "gauge_id": gid,
            "water_body": row["water_body"],
            "regulation_tier": row["regulation_tier"],
            "n_days": len(raw),
            "pct_complete_raw": round(pct, 2),
            "pct_usable_final": round(100 * out["waterlevel_m"].notna().mean(), 2),
            "n_spike": int(counts.get("spike", 0)),
            "n_flatline": int(counts.get("flatline", 0)),
            "n_interp_short": int(counts.get("interp_short", 0)),
            "n_interp_seasonal": int(counts.get("interp_seasonal", 0)),
            "n_gap_long": int(counts.get("gap_long", 0)),
            "start": str(raw.index.min().date()),
            "end": str(raw.index.max().date()),
        })
        print(f"    {gid} {row['water_body'][:24]:<24} {row['regulation_tier']:<3} "
              f"{pct:5.1f}%  spikes={counts.get('spike', 0):<4} "
              f"flat={counts.get('flatline', 0)}")

    if not frames:
        raise RuntimeError("no lakes passed the completeness filter")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(qc_rows)


# --------------------------------------------------------- 7. data dictionary
def data_dictionary(daily, static):
    defs = {
        "gauge_id": ("BAFU station ID", "-", "CAMELS-CH stations"),
        "water_body": ("Lake name", "-", "CAMELS-CH stations"),
        "regulation_tier": ("N0/N1/R natural classification", "-", "DERIVED from human-influence attributes"),
        "date": ("Observation date", "date", "CAMELS-CH observation_based"),
        "waterlevel_raw_m": ("Water level as published", "m", "CAMELS-CH / BAFU"),
        "waterlevel_m": ("Water level after QC and gap handling", "m", "DERIVED"),
        "qc_flag": ("ok/spike/flatline/interp_short/interp_seasonal/gap_long", "-", "DERIVED"),
        "precipitation_mm": ("Catchment daily precipitation", "mm/d", "CAMELS-CH / MeteoSwiss"),
        "temperature_mean_c": ("Catchment mean air temperature", "degC", "CAMELS-CH / MeteoSwiss"),
        "swe_mm": ("Snow water equivalent", "mm", "CAMELS-CH / SLF"),
        "api_30": ("Antecedent precipitation index, 30d halflife", "mm", "DERIVED"),
        "api_60": ("Antecedent precipitation index, 60d halflife", "mm", "DERIVED"),
        "api_90": ("Antecedent precipitation index, 90d halflife", "mm", "DERIVED"),
        "snowmelt_proxy": ("SWE decrease on days above 0 degC", "mm/d", "DERIVED"),
        "snowmelt_cum30": ("30-day cumulative snowmelt proxy", "mm", "DERIVED"),
        "degree_days": ("Mean temperature floored at zero", "degC", "DERIVED"),
        "cdd_30": ("30-day cumulative degree-days", "degC.d", "DERIVED"),
        "level_delta1": ("Day-on-day level change", "m", "DERIVED"),
        "doy_sin": ("Cyclical day-of-year, sine", "-", "DERIVED"),
        "doy_cos": ("Cyclical day-of-year, cosine", "-", "DERIVED"),
        "specific_storage_mm": ("Upstream reservoir capacity per catchment area", "mm", "DERIVED"),
        "catchment_area_km2": ("Catchment area", "km2", "CAMELS-CH topographic"),
    }
    rows = []
    for table, df in (("lakes_daily", daily), ("lakes_static", static)):
        for c in df.columns:
            if c in defs:
                desc, unit, src = defs[c]
            elif re.match(r"level_lag\d+", c):
                desc, unit, src = (f"Water level lagged {c.split('lag')[1]} days", "m", "DERIVED")
            elif re.match(r"level_roll\d+", c):
                desc, unit, src = (f"Rolling {c} of past levels", "m", "DERIVED")
            else:
                desc, unit, src = ("See CAMELS-CH documentation", "-", "CAMELS-CH")
            rows.append({"table": table, "column": c, "description": desc,
                         "unit": unit, "source": src,
                         "dtype": str(df[c].dtype)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- 8. driver
def build(root):
    root = Path(root).expanduser().resolve()
    OUT.mkdir(exist_ok=True)

    lakes = select_lakes(root)
    static = classify_regulation(root, lakes)
    static = attach_attributes(root, static)
    daily, qc = build_panel(root, static)

    static = static[static["gauge_id"].isin(daily["gauge_id"].unique())].reset_index(drop=True)
    static = static.merge(
        qc[["gauge_id", "pct_complete_raw", "pct_usable_final", "start", "end"]],
        on="gauge_id", how="left")

    dd = data_dictionary(daily, static)

    rule("6. WRITING OUTPUTS")
    static.to_csv(OUT / "lakes_static.csv", index=False)
    daily.to_csv(OUT / "lakes_daily.csv", index=False)
    dd.to_csv(OUT / "data_dictionary.csv", index=False)
    qc.to_csv(OUT / "qc_report.csv", index=False)

    xl = OUT / "natural_lakes.xlsx"
    if len(daily) < 1_000_000:
        with pd.ExcelWriter(xl, engine="openpyxl") as w:
            static.to_excel(w, sheet_name="lakes_static", index=False)
            daily.to_excel(w, sheet_name="lakes_daily", index=False)
            qc.to_excel(w, sheet_name="qc_report", index=False)
            dd.to_excel(w, sheet_name="data_dictionary", index=False)
        print(f"    {xl}")
    else:
        print(f"    !! {len(daily):,} rows exceeds the Excel limit; CSVs only")

    for f in ("lakes_static.csv", "lakes_daily.csv", "data_dictionary.csv", "qc_report.csv"):
        print(f"    {OUT / f}")

    rule("SUMMARY")
    print(f"    lakes retained : {static['gauge_id'].nunique()}")
    print(f"    daily rows     : {len(daily):,}")
    print(f"    date range     : {daily['date'].min().date()} to {daily['date'].max().date()}")
    print(f"    features       : {daily.shape[1]} columns")
    print("\n" + static["regulation_tier"].value_counts().to_string())
    print("\n    N0 lakes (your strictly natural primary sample):")
    for _, r in static[static["regulation_tier"] == "N0"].iterrows():
        print(f"      {int(r['gauge_id'])}  {r['water_body']}")
    return static, daily, qc, dd


def _cli():
    """Only meaningful from a terminal. In Jupyter, call build(path) directly."""
    in_notebook = any("ipykernel" in a or a.endswith(".json") for a in sys.argv)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if in_notebook or not args:
        print("Functions loaded. Now run:")
        print("    static, daily, qc, dd = build(r'C:\\path\\to\\camels_ch')")
        return
    build(args[0])


if __name__ == "__main__":
    _cli()
