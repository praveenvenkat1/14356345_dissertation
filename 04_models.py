"""
04_models.py
forecasts lake level at h = 1, 3, 7 days
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT = Path("output")
FIG = Path("figures")

SEQ_LEN = 90
HORIZONS = (1, 3, 7)
TRAIN_END = "2010-12-31"
VAL_END = "2015-12-31"

DYNAMIC = [
    "waterlevel_m", "precipitation_mm", "temperature_mean_c", "swe_mm",
    "snowmelt_proxy", "api_30", "api_60", "cdd_30", "level_delta1",
    "doy_sin", "doy_cos",
]
STATIC_PREF = ["mean_elevation_m", "mean_slope", "frac_snow", "glac_area",
               "catchment_area_km2", "p_mean", "aridity"]
TARGET = "waterlevel_m"

EPOCHS = 40
PATIENCE = 6
BATCH = 512
HIDDEN = 64
EMB_DIM = 8
LR = 1e-3


def rule(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


# ------------------------------------------------------------------- metrics
def nse(y, yhat):
    """Nash-Sutcliffe efficiency. 1 is perfect, 0 equals predicting the mean."""
    denom = np.sum((y - np.mean(y)) ** 2)
    return np.nan if denom == 0 else 1 - np.sum((y - yhat) ** 2) / denom


def kge(y, yhat):
    """Kling-Gupta efficiency: correlation, variance ratio, bias ratio."""
    if np.std(y) == 0 or np.std(yhat) == 0 or np.mean(y) == 0:
        return np.nan
    r = np.corrcoef(y, yhat)[0, 1]
    alpha = np.std(yhat) / np.std(y)
    beta = np.mean(yhat) / np.mean(y)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def score(y, yhat, y_persist=None):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    if len(y) < 10:
        return {k: np.nan for k in ("rmse", "mae", "nse", "kge", "pss", "n")}
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    out = {"rmse": rmse, "mae": float(np.mean(np.abs(y - yhat))),
           "nse": nse(y, yhat), "kge": kge(y, yhat), "n": int(len(y))}
    if y_persist is not None:
        yp = np.asarray(y_persist, float)[m]
        rmse_p = float(np.sqrt(np.mean((y - yp) ** 2)))
        out["pss"] = np.nan if rmse_p == 0 else 1 - rmse / rmse_p
    else:
        out["pss"] = np.nan
    return out


# --------------------------------------------------------------- preparation
def prepare(daily, static):
    """Per-lake arrays, temporal masks, and train-only scaling."""
    rule("PREPARING DATA")
    stat_cols = [c for c in STATIC_PREF if c in static.columns
                 and static[c].notna().any()]
    print(f"    static attributes: {stat_cols}")

    dyn_cols = [c for c in DYNAMIC if c in daily.columns]
    missing = set(DYNAMIC) - set(dyn_cols)
    if missing:
        print(f"    !! absent, skipped: {sorted(missing)}")

    gids = sorted(static["gauge_id"].unique())
    lake_index = {g: i for i, g in enumerate(gids)}

    # gather training rows first so scalers never see validation or test data
    tr_parts = []
    for g in gids:
        d = daily[daily["gauge_id"] == g]
        tr_parts.append(d[d["date"] <= TRAIN_END][dyn_cols])
    tr = pd.concat(tr_parts)
    drv = [c for c in dyn_cols if c != TARGET]
    drv_mu, drv_sd = tr[drv].mean(), tr[drv].std().replace(0, 1.0)

    sm = static.set_index("gauge_id")[stat_cols].apply(pd.to_numeric, errors="coerce")
    sm = (sm - sm.mean()) / sm.std().replace(0, 1.0)
    sm = sm.fillna(0.0)

    lakes = {}
    for g in gids:
        d = daily[daily["gauge_id"] == g].sort_values("date").reset_index(drop=True)
        if len(d) < SEQ_LEN + 400:
            continue
        lvl_tr = d.loc[d["date"] <= TRAIN_END, TARGET]
        mu, sd = lvl_tr.mean(), lvl_tr.std()
        if not np.isfinite(sd) or sd == 0:
            continue

        X = pd.DataFrame(index=d.index)
        X[TARGET] = (d[TARGET] - mu) / sd            # per-lake standardisation
        for c in drv:
            X[c] = (d[c] - drv_mu[c]) / drv_sd[c]
        X = X[dyn_cols].astype("float32")

        lakes[g] = {
            "dates": d["date"].values,
            "X": np.nan_to_num(X.to_numpy(), nan=0.0),
            "valid": X.notna().all(axis=1).to_numpy() & d[TARGET].notna().to_numpy(),
            "y_raw": d[TARGET].to_numpy(dtype="float64"),
            "mu": float(mu), "sd": float(sd),
            "idx": lake_index[g],
            "static": sm.loc[g].to_numpy(dtype="float32") if g in sm.index
            else np.zeros(len(stat_cols), "float32"),
            "name": static.loc[static["gauge_id"] == g, "water_body"].iloc[0],
        }

    print(f"    lakes prepared: {len(lakes)}")
    return lakes, dyn_cols, stat_cols


def windows(lakes, horizon):
    """Index every usable (lake, t) window. Split by the date of the target."""
    sets = {"train": [], "val": [], "test": []}
    tr_end, va_end = np.datetime64(TRAIN_END), np.datetime64(VAL_END)
    for g, L in lakes.items():
        n = len(L["dates"])
        for t in range(SEQ_LEN - 1, n - horizon):
            if not L["valid"][t - SEQ_LEN + 1: t + 1].all():
                continue
            if not np.isfinite(L["y_raw"][t + horizon]):
                continue
            td = L["dates"][t + horizon]
            key = "train" if td <= tr_end else ("val" if td <= va_end else "test")
            sets[key].append((g, t))
    print(f"    h={horizon}: train {len(sets['train']):,}  "
          f"val {len(sets['val']):,}  test {len(sets['test']):,}")
    return sets


# ------------------------------------------------------------------ baselines
def baselines(lakes, sets, horizon):
    rows = []
    for g, L in lakes.items():
        idx = [t for gg, t in sets["test"] if gg == g]
        if not idx:
            continue
        idx = np.array(idx)
        y = L["y_raw"][idx + horizon]
        persist = L["y_raw"][idx]

        # climatology from training years only
        dts = pd.to_datetime(L["dates"])
        tr = dts <= pd.Timestamp(TRAIN_END)
        clim = pd.Series(L["y_raw"][tr]).groupby(dts[tr].dayofyear).mean()
        doy_t = dts[idx + horizon].dayofyear
        climo = np.array([clim.get(d, np.nan) for d in doy_t])

        for name, pred in (("persistence", persist), ("climatology", climo)):
            s = score(y, pred, persist)
            s.update(model=name, horizon=horizon, gauge_id=g, water_body=L["name"])
            rows.append(s)
    return rows


# ----------------------------------------------------- tabular feature models
def tabular(lakes, sets, horizon, stat_cols):
    """Ridge and LightGBM on flattened recent-window features."""
    from sklearn.linear_model import Ridge

    def featurise(pairs):
        Xs, ys, meta = [], [], []
        for g, t in pairs:
            L = lakes[g]
            w = L["X"][t - SEQ_LEN + 1: t + 1]
            f = np.concatenate([
                w[-1], w[-7:].mean(0), w[-30:].mean(0), w[-90:].mean(0),
                w[-30:].std(0), L["static"],
            ])
            Xs.append(f)
            ys.append((L["y_raw"][t + horizon] - L["mu"]) / L["sd"])
            meta.append((g, t))
        return np.asarray(Xs, "float32"), np.asarray(ys, "float32"), meta

    Xtr, ytr, _ = featurise(sets["train"])
    Xte, yte, mte = featurise(sets["test"])
    print(f"    feature matrix: {Xtr.shape}")

    preds = {}
    preds["ridge"] = Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte)
    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                              num_leaves=31, verbose=-1)
        preds["lightgbm"] = m.fit(Xtr, ytr).predict(Xte)
    except ImportError:
        print("    lightgbm not installed - skipped (pip install lightgbm)")

    return preds, mte


# ----------------------------------------------------------------- the LSTM
def lstm(lakes, sets, horizon, n_static):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("    torch not installed - LSTM skipped.")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return None, None

    torch.manual_seed(0)
    np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    device: {dev}")

    n_feat = next(iter(lakes.values()))["X"].shape[1]
    n_lakes = max(L["idx"] for L in lakes.values()) + 1

    class DS(torch.utils.data.Dataset):
        """Slices windows on demand - materialising them all would be enormous."""
        def __init__(self, pairs):
            self.pairs = pairs

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, i):
            g, t = self.pairs[i]
            L = lakes[g]
            return (torch.from_numpy(L["X"][t - SEQ_LEN + 1: t + 1]),
                    torch.tensor(L["idx"]),
                    torch.from_numpy(L["static"]),
                    torch.tensor((L["y_raw"][t + horizon] - L["mu"]) / L["sd"],
                                 dtype=torch.float32))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(n_lakes, EMB_DIM)
            self.rnn = nn.LSTM(n_feat, HIDDEN, batch_first=True)
            self.drop = nn.Dropout(0.2)
            self.head = nn.Sequential(
                nn.Linear(HIDDEN + EMB_DIM + n_static, 64), nn.ReLU(),
                nn.Linear(64, 1))

        def forward(self, x, lid, st):
            _, (h, _) = self.rnn(x)
            z = torch.cat([self.drop(h[-1]), self.emb(lid), st], dim=1)
            return self.head(z).squeeze(1)

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lossf = nn.MSELoss()

    dl_tr = torch.utils.data.DataLoader(DS(sets["train"]), batch_size=BATCH,
                                        shuffle=True, drop_last=True)
    dl_va = torch.utils.data.DataLoader(DS(sets["val"]), batch_size=BATCH)
    dl_te = torch.utils.data.DataLoader(DS(sets["test"]), batch_size=BATCH)

    best, bad, best_state = np.inf, 0, None
    for ep in range(1, EPOCHS + 1):
        net.train()
        for x, lid, st, y in dl_tr:
            opt.zero_grad()
            loss = lossf(net(x.to(dev), lid.to(dev), st.to(dev)), y.to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        net.eval()
        vl, nb = 0.0, 0
        with torch.no_grad():
            for x, lid, st, y in dl_va:
                vl += lossf(net(x.to(dev), lid.to(dev), st.to(dev)),
                            y.to(dev)).item()
                nb += 1
        vl /= max(nb, 1)
        print(f"      epoch {ep:>2}  val MSE {vl:.5f}"
              + ("  <- best" if vl < best else ""))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"      early stop at epoch {ep}")
                break

    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    out = []
    with torch.no_grad():
        for x, lid, st, _ in dl_te:
            out.append(net(x.to(dev), lid.to(dev), st.to(dev)).cpu().numpy())
    return np.concatenate(out), sets["test"]


# ------------------------------------------------------------ score predictions
def score_preds(lakes, preds, meta, horizon, model_name):
    """De-standardise per lake, then score per lake and by season and extreme."""
    by_lake = {}
    for (g, t), p in zip(meta, preds):
        by_lake.setdefault(g, ([], []))
        by_lake[g][0].append(t)
        by_lake[g][1].append(p)

    rows, strat = [], []
    for g, (ts, ps) in by_lake.items():
        L = lakes[g]
        ts = np.array(ts)
        yhat = np.array(ps) * L["sd"] + L["mu"]
        y = L["y_raw"][ts + horizon]
        persist = L["y_raw"][ts]

        s = score(y, yhat, persist)
        s.update(model=model_name, horizon=horizon, gauge_id=g, water_body=L["name"])
        rows.append(s)

        # seasonal breakdown - works even at n=14, since it splits within lakes
        month = pd.to_datetime(L["dates"][ts + horizon]).month
        seasons = {"winter": [12, 1, 2], "snowmelt": [3, 4, 5],
                   "summer": [6, 7, 8], "autumn": [9, 10, 11]}
        for nm, mo in seasons.items():
            m = np.isin(month, mo)
            if m.sum() > 30:
                ss = score(y[m], yhat[m], persist[m])
                ss.update(model=model_name, horizon=horizon, gauge_id=g,
                          water_body=L["name"], stratum=f"season_{nm}")
                strat.append(ss)

        # extremes: the flood and drought tails
        for nm, m in (("high_5pct", y >= np.nanpercentile(y, 95)),
                      ("low_5pct", y <= np.nanpercentile(y, 5))):
            if m.sum() > 30:
                ss = score(y[m], yhat[m], persist[m])
                ss.update(model=model_name, horizon=horizon, gauge_id=g,
                          water_body=L["name"], stratum=nm)
                strat.append(ss)
    return rows, strat


# -------------------------------------------------------------------- driver
def run_models(sample_csv="output/sample_N0.csv",
               daily_csv="output/lakes_daily.csv", tag="N0"):
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    static = pd.read_csv(sample_csv)
    daily = pd.read_csv(daily_csv, parse_dates=["date"])
    daily = daily[daily["gauge_id"].isin(static["gauge_id"])]

    lakes, dyn_cols, stat_cols = prepare(daily, static)
    all_rows, all_strat = [], []

    for h in HORIZONS:
        rule(f"HORIZON h = {h} DAYS")
        sets = windows(lakes, h)
        if len(sets["train"]) < 500 or len(sets["test"]) < 100:
            print("    too few windows - skipped")
            continue

        all_rows += baselines(lakes, sets, h)

        print("    fitting tabular models...")
        tp, meta = tabular(lakes, sets, h, stat_cols)
        for name, p in tp.items():
            r, s = score_preds(lakes, p, meta, h, name)
            all_rows += r
            all_strat += s

        print("    training LSTM...")
        p, meta_l = lstm(lakes, sets, h, len(stat_cols))
        if p is not None:
            r, s = score_preds(lakes, p, meta_l, h, "lstm")
            all_rows += r
            all_strat += s

    res = pd.DataFrame(all_rows)
    strat = pd.DataFrame(all_strat)
    res.to_csv(OUT / f"model_results_{tag}.csv", index=False)
    if len(strat):
        strat.to_csv(OUT / f"model_strata_{tag}.csv", index=False)

    rule("RESULTS: MEDIAN ACROSS LAKES")
    piv = res.pivot_table(index="model", columns="horizon",
                          values=["rmse", "nse", "pss"], aggfunc="median")
    print(piv.round(4).to_string())
    print("\n    pss > 0 means the model beats persistence. That is the claim"
          "\n    that matters; NSE alone can look excellent while losing to it.")

    (OUT / f"model_config_{tag}.json").write_text(json.dumps({
        "seq_len": SEQ_LEN, "horizons": list(HORIZONS), "train_end": TRAIN_END,
        "val_end": VAL_END, "hidden": HIDDEN, "emb_dim": EMB_DIM,
        "batch": BATCH, "epochs": EPOCHS, "patience": PATIENCE, "lr": LR,
        "dynamic_features": dyn_cols, "static_features": stat_cols,
        "n_lakes": len(lakes),
    }, indent=2), encoding="utf-8")

    print(f"\n    -> {OUT / f'model_results_{tag}.csv'}")
    print(f"    -> {OUT / f'model_strata_{tag}.csv'}")
    print(f"    -> {OUT / f'model_config_{tag}.json'}   (paste this into Methodology)")
    return {"results": res, "strata": strat, "lakes": lakes}