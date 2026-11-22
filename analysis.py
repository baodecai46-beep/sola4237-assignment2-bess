# %% [markdown]
# # SOLA4237 Assignment 2: household BESS optimisation (Shepparton, VIC)
#
# Pipeline (run top to bottom, `python analysis.py`):
#   1. Load, check and clean 5-minute telemetry (2023 = year 1, 2024 = year 2)
#   2. Site analysis (load vs PV, overnight energy need -> battery sizing)
#   3. PV model trained on 2023, tested on 2024 against the commercial forecast
#   4. Wholesale price forecast model (48 h ahead, issued hourly)
#   5. Battery model (AC- vs DC-coupled, SoC limits, temperature derating, degradation cost)
#   6. Perfect-foresight optimisation (LP) for flat, ToU and wholesale passthrough tariffs
#   7. Battery sizing sweep and payback
#   8. 48 h rolling-horizon scheduler with forecasts, 5-minute execution with actual values
#   9. Techno-economics of the VPP product, sensitivities, figures, validation checks
#
# Sign convention: all powers in kW (AC side) and positive; energy = power x 5/60 h.
# Battery: charge c >= 0, discharge d >= 0. Grid: import gi >= 0, export ge >= 0.

# %%
import json
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA = HERE / "data"
RES = HERE / "results"
RES.mkdir(exist_ok=True)
RAW_CSV = DATA / "Data_VIC_Shepparton - Hot Dry Summer, Cool Winter_2023-2024_partial.csv"
DT = 5 / 60  # hours per interval
STEPS_PER_HOUR = 12
STEPS_PER_DAY = 288
HORIZON = 48 * STEPS_PER_HOUR  # 576 five-minute steps
LOG = []


def pmap(func, items):
    """Parallel map with the 'fork' start method (works in scripts and notebooks on Linux/macOS);
    falls back to a plain loop on Windows or if N_JOBS=1."""
    n = int(os.environ.get("N_JOBS", "4"))
    if n > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(n) as pool:
            return pool.map(func, items)
    return [func(x) for x in items]


def log(msg):
    print(msg)
    LOG.append(str(msg))


def check(cond, msg):
    """Validation assertion that is also written to the validation log."""
    assert cond, msg
    LOG.append("PASS: " + msg)


# %% [markdown]
# ## 0. Assumptions (all in one place, saved to results/assumptions.json)

# %%
A = dict(
    # PV (metadata of the CER data generator; same generator as Assignment 1)
    pv_gamma=-0.004,          # 1/degC temperature coefficient
    pv_pr=0.85,               # performance ratio (used only to translate fitted kW scale to kWp)
    noct=45.0,                # degC, nominal operating cell temperature
    # Battery base case: AC-coupled retrofit (existing PV inverter kept)
    bess_kwh=10.0,            # nominal capacity, chosen by sizing sweep (section 7)
    bess_kw=5.0,              # inverter charge/discharge limit
    soc_min=0.10, soc_max=0.95,
    eta_c_pv=0.95, eta_c_grid=0.95, eta_d=0.95,  # AC-coupled: ~90% round trip
    deg_cost=0.05,            # $/kWh discharged (wear cost used inside the optimiser)
    derate_hot_start=35.0,    # degC: linear derate above, 5%/degC, floor 50%
    cold_charge_limit_c=5.0,  # degC: charge limited to 50% below this
    grid_kw=15.0,             # connection limit (63 A single phase approx.)
    # DC-coupled alternative (hybrid inverter): PV->battery avoids one AC conversion
    dc_eta_c_pv=0.975, dc_eta_c_grid=0.93, dc_eta_d=0.95,
    # Wholesale passthrough adders, AUD/kWh excl. GST (ASSUMED, see report)
    net_peak=0.16, net_off=0.06, net_peak_hours=(15, 21),  # Powercor-style residential ToU network
    env_cert=0.020,           # LRET + SRES + VEU certificate costs
    market_fees=0.002,        # AEMO fees and ancillary services
    hedging=0.015,            # retailer risk/hedging and margin
    vpp_fee_month=25.0,       # $/month VPP-SPREE fee (brief)
    rt_spike=0.30,            # $/kWh: real-time override threshold (last settled 5-min price)
    # Economics
    capex_fixed=2000.0, capex_per_kwh=750.0,  # installed AC-coupled battery, 2024 AUD (ASSUMED)
    rebate_per_kwh=300.0,     # Cheaper Home Batteries rebate sensitivity (approx., verify)
    discount_rate=0.05, life_years=10, fade_per_year=0.02,
)
json.dump(A, open(RES / "assumptions.json", "w"), indent=2)

# %% [markdown]
# ## 1. Load, check and clean telemetry

# %%
raw = pd.read_csv(RAW_CSV, parse_dates=["timestamp"])
raw = raw.rename(columns={
    "irradiance_W/m^2": "irr", "temperature_C": "temp", "solar_output_kW": "pv_meas",
    "commercial_forecasted_solar_output_kW": "pv_comm", "load_kW": "load",
    "spot_price_AUD/kWh": "spot", "tariff_import_flat_AUD/kWh": "imp_flat",
    "tariff_export_flat_AUD/kWh": "exp_flat", "tariff_import_tou_AUD/kWh": "imp_tou",
    "tariff_export_tou_AUD/kWh": "exp_tou"})
df = raw.set_index("timestamp").sort_index()
full_index = pd.date_range("2023-01-01 00:00", "2024-12-31 23:55", freq="5min")
check(df.index.is_unique, "timestamps unique (no DST duplicates)")
check(len(df) == len(full_index) and (df.index == full_index).all(),
      "regular 5-minute grid, 2023-01-01 to 2024-12-31, no missing rows")
y1, y2 = df.index.year == 2023, df.index.year == 2024

# Time convention: the irradiance-weighted centroid ("solar noon") stays within 12:30-13:00 in every
# month with no one-hour jump at the DST changeovers -> fixed clock (AEST/NEM time), no daylight saving.
centroid = {}
for mo in range(1, 13):
    s = df.loc[(df.index.year == 2023) & (df.index.month == mo)]
    h = s.index.hour + s.index.minute / 60
    centroid[mo] = float((s["irr"] * h).sum() / s["irr"].sum())
log(f"Monthly irradiance centroid (h): {np.round(list(centroid.values()), 2)} -> no DST shift, fixed AEST")
check(max(centroid.values()) - min(centroid.values()) < 0.6, "no daylight-saving shift in timestamps")

# Missing-data audit
miss = {}
for c in ["irr", "temp", "load"]:
    miss[c] = {"2023": int(df.loc[y1, c].isna().sum()), "2024": int(df.loc[y2, c].isna().sum())}
miss["pv_meas"] = {"2023": int(df.loc[y1, "pv_meas"].isna().sum()), "2024": "not provided"}
miss["pv_comm"] = {"2023": "not provided", "2024": int(df.loc[y2, "pv_comm"].isna().sum())}
miss["spot"] = {"2023": int(df.loc[y1, "spot"].isna().sum()), "2024": "not provided (AEMO)"}


def gap_runs(s):
    m = s.isna()
    g = (m != m.shift()).cumsum()
    r = m.groupby(g).agg(["first", "size"])
    return r.loc[r["first"], "size"]


gr = gap_runs(df["load"])
miss["load_gap_runs"] = {"count": int(len(gr)), "max_steps": int(gr.max()),
                         "longest_gap_day": str(df.index[df["load"].isna()].to_series().dt.date.value_counts().idxmax())}
check((df["irr"].isna() == df["load"].isna()).all() and (df["temp"].isna() == df["load"].isna()).all(),
      "irradiance, temperature and load gaps occur at identical timestamps (logger outage)")
log(f"Missing data: {miss}")

# Spurious-data checks (flag; only physically impossible values are changed)
flags = {}
flags["negative_values"] = int((df[["irr", "pv_meas", "pv_comm", "load"]] < 0).sum().sum())
flags["pv_at_night"] = int(((df["irr"] == 0) & (df["pv_meas"] > 0.05)).sum())
flags["irr_above_1100"] = int((df["irr"] > 1100).sum())
t = df["temp"]
run_id = (t != t.shift()).cumsum()
run_len = t.groupby(run_id).transform("size")
flags["temp_flatline_steps_ge_3h"] = int(((run_len >= 36) & t.notna()).sum())
flags["temp_exact_zero_2023"] = int((df.loc[y1, "temp"] == 0).sum())
flags["temp_min_2023"] = float(df.loc[y1, "temp"].min())
flags["temp_min_2024"] = float(df.loc[y2, "temp"].min())
flags["spot_outside_market_limits"] = int(((df["spot"] < -1.0) | (df["spot"] > 16.6)).sum())
flags["spot_negative_share_2023"] = float((df.loc[y1, "spot"] < 0).mean())
flags["spot_max_2023"] = float(df.loc[y1, "spot"].max())
check(flags["negative_values"] == 0 and flags["pv_at_night"] == 0, "no negative power or night-time PV")
check(flags["spot_outside_market_limits"] == 0, "2023 spot prices within NEM market floor/cap")


def fill_series(s, max_interp=12, window_days=7):
    """Fill gaps: linear interpolation for gaps <= 1 h, otherwise the mean of the same
    5-minute slot over a centred +/-3-day window (keeps the daily shape)."""
    s = s.copy()
    s = s.interpolate(limit=max_interp, limit_area="inside")
    if s.isna().any():
        day = s.index.normalize()
        slot = (s.index.hour * 60 + s.index.minute) // 5
        tab = pd.DataFrame({"v": s.values, "day": day, "slot": slot}).pivot(index="day", columns="slot", values="v")
        prof = tab.rolling(window_days, center=True, min_periods=1).mean()
        prof = prof.ffill().bfill()
        lookup = prof.stack().reindex(list(zip(day, slot))).values
        s = s.fillna(pd.Series(lookup, index=s.index))
    return s


clean = pd.DataFrame(index=df.index)
for c in ["irr", "temp", "load"]:
    clean[c] = fill_series(df[c])
clean["irr"] = clean["irr"].clip(lower=0)
clean["imp_flat"], clean["exp_flat"] = df["imp_flat"], df["exp_flat"]
clean["imp_tou"], clean["exp_tou"] = df["imp_tou"], df["exp_tou"]
clean["filled"] = df["load"].isna()
check(clean[["irr", "temp", "load"]].notna().all().all(), "load/weather complete after gap filling")
check(abs(df["load"].mean() - clean["load"].mean()) / df["load"].mean() < 0.01,
      "gap filling changes mean load by <1%")

# %% [markdown]
# ## 2. PV model (train on 2023 measured PV, test on 2024 vs commercial forecast)

# %%
def cell_temp(irr, temp):
    return temp + irr / 800.0 * (A["noct"] - 20.0)


def phys_basis(irr, temp):
    """Plane-of-array proxy x temperature correction; PV = k * basis."""
    return irr / 1000.0 * (1 + A["pv_gamma"] * (cell_temp(irr, temp) - 25.0))


def pv_features(frame):
    idx = frame.index
    hod = idx.hour + idx.minute / 60
    doy = idx.dayofyear
    return pd.DataFrame({
        "irr": frame["irr"].values, "temp": frame["temp"].values,
        "basis": phys_basis(frame["irr"], frame["temp"]).values,
        "hs": np.sin(2 * np.pi * hod / 24), "hc": np.cos(2 * np.pi * hod / 24),
        "ds": np.sin(2 * np.pi * doy / 365.25), "dc": np.cos(2 * np.pi * doy / 365.25)}, index=idx)


train_mask = y1 & df["pv_meas"].notna() & df["irr"].notna()
tr = df.loc[train_mask]
b = phys_basis(tr["irr"], tr["temp"])
k_fit = float((b * tr["pv_meas"]).sum() / (b * b).sum())  # least squares through origin
kwp_est = k_fit / A["pv_pr"]
log(f"Physical PV model: P = {k_fit:.2f} kW x G/1000 x temp factor -> ~{kwp_est:.1f} kWp at PR {A['pv_pr']}")

Xtr = pv_features(tr)
pv_ml = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
day_mask_tr = tr["irr"] > 0
pv_ml.fit(Xtr[day_mask_tr], tr.loc[day_mask_tr, "pv_meas"])


def pv_predict(frame, kind="ml"):
    if kind == "phys":
        return (k_fit * phys_basis(frame["irr"], frame["temp"])).clip(lower=0)
    out = np.zeros(len(frame))
    m = frame["irr"].values > 0
    out[m] = pv_ml.predict(pv_features(frame)[m])
    return pd.Series(np.clip(out, 0, None), index=frame.index)


clean["pv_phys"] = pv_predict(clean, "phys")
clean["pv_ml"] = pv_predict(clean, "ml")

# 2023: fill missing measured PV with the ML model (irradiance available after filling)
pv1 = df["pv_meas"].copy()
pv1[y1 & pv1.isna()] = clean.loc[y1 & pv1.isna(), "pv_ml"]
# "Actual" PV: measured in 2023, commercial forecast treated as measurement in 2024 (brief)
clean["pv"] = np.where(y1, pv1, df["pv_comm"])
check(clean["pv"].notna().all(), "PV series complete for both years")

# Naive weather-free benchmark: same time yesterday (persistence)
clean["pv_persist"] = clean["pv"].shift(STEPS_PER_DAY).bfill()


def err_metrics(pred, act, cap):
    e = pred - act
    day = act.index.hour.isin(range(6, 20))
    return dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))),
                nRMSE_cap_pct=float(100 * np.sqrt(np.mean(e ** 2)) / cap),
                nRMSE_daymean_pct=float(100 * np.sqrt(np.mean(e[day] ** 2)) / act[day].mean()),
                bias=float(np.mean(e)), R2=float(1 - np.sum(e ** 2) / np.sum((act - act.mean()) ** 2)),
                annual_kWh_pred=float(pred.sum() * DT), annual_kWh_act=float(act.sum() * DT))


cap = kwp_est
pv_metrics = {}
tr_act = tr["pv_meas"]
pv_metrics["phys_train_2023"] = err_metrics(clean.loc[train_mask, "pv_phys"], tr_act, cap)
pv_metrics["ml_train_2023"] = err_metrics(clean.loc[train_mask, "pv_ml"], tr_act, cap)
act24 = clean.loc[y2, "pv"]
for name in ["pv_phys", "pv_ml", "pv_persist"]:
    pv_metrics[name + "_test_2024_vs_commercial"] = err_metrics(clean.loc[y2, name], act24, cap)
pd.DataFrame(pv_metrics).T.to_csv(RES / "pv_model_metrics.csv")
log(pd.DataFrame(pv_metrics).T.round(3).to_string())
PV_FC = "pv_ml" if pv_metrics["pv_ml_test_2024_vs_commercial"]["RMSE"] <= \
    pv_metrics["pv_phys_test_2024_vs_commercial"]["RMSE"] else "pv_phys"
log(f"PV forecast used in the scheduler: {PV_FC}")

# %% [markdown]
# ## 3. Wholesale prices for 2024
# The supplied CSV has no 2024 spot price. Official AEMO VIC1 5-minute RRP files
# (PRICE_AND_DEMAND_2024MM_VIC1.csv) are joined exactly on SETTLEMENTDATE labels.
# If they are absent, a clearly-flagged PLACEHOLDER (2023 prices shifted by 52 weeks) is used.

# %%
files = []
for aemo_dir in [DATA / "aemo", HERE.parent / "data" / "aemo", HERE.parent]:  # package or assignment folder
    files = sorted(aemo_dir.glob("PRICE_AND_DEMAND_2024*_VIC1.csv")) + sorted(aemo_dir.glob("PRICE_AND_DEMAND_202501_VIC1.csv"))
    if len(files) >= 12:
        break
spot = df["spot"].copy()
if len(files) >= 12:
    a = pd.concat([pd.read_csv(f) for f in files])
    a["ts"] = pd.to_datetime(a["SETTLEMENTDATE"])
    a = a.drop_duplicates("ts").set_index("ts")["RRP"] / 1000.0
    spot.loc[y2] = a.reindex(df.index[y2]).values
    PRICE_SOURCE = "AEMO VIC1 2024 RRP (official)"
    n_missing = int(spot.loc[y2].isna().sum())
    spot.loc[y2] = spot.loc[y2].interpolate(limit_direction="both")
    log(f"Joined AEMO 2024 prices, {n_missing} labels missing and interpolated")
else:
    shifted = df["spot"].shift(364 * STEPS_PER_DAY)  # same weekday one year later
    spot.loc[y2] = shifted.loc[y2]
    spot.loc[y2] = spot.loc[y2].fillna(df["spot"].shift(371 * STEPS_PER_DAY).loc[y2])
    PRICE_SOURCE = "PLACEHOLDER: 2023 VIC1 prices shifted 52 weeks (AEMO 2024 files not supplied)"
log(f"2024 price source: {PRICE_SOURCE}")
clean["spot"] = spot
check(clean["spot"].notna().all(), "spot price complete for both years")

hod = clean.index.hour
net = np.where((hod >= A["net_peak_hours"][0]) & (hod < A["net_peak_hours"][1]), A["net_peak"], A["net_off"])
adders = net + A["env_cert"] + A["market_fees"] + A["hedging"]
clean["imp_ws"] = clean["spot"] + adders
clean["exp_ws"] = clean["spot"]

# %% [markdown]
# ## 4. Price forecast (issued every hour, 48 hourly values)
# Gradient-boosted trees on: horizon, target hour/weekday/month, last observed hourly price,
# last-24 h mean, most recent same-hour price, 7-day median at the target hour,
# target-hour temperature and irradiance (weather forecast assumed accurate).

# %%
hp = clean["spot"].resample("h").mean()
hw = clean[["temp", "irr"]].resample("h").mean()


def price_design(issue_times):
    """Rows = (issue time, horizon 1..48). Horizon 1 is the hour that starts at the issue time,
    so its price is not yet known when the battery bids."""
    pidx = {t: i for i, t in enumerate(hp.index)}
    vals = hp.values
    hod_arr = hp.index.hour.values
    rows = []
    for t0 in issue_times:
        i0 = pidx[t0]
        if i0 < 24 * 7:
            continue
        last = vals[i0 - 1]
        m24 = vals[i0 - 24:i0].mean()
        hist = vals[i0 - 24 * 7:i0]
        hist_hod = hod_arr[i0 - 24 * 7:i0]
        for h in range(1, 49):
            j = i0 + h - 1
            if j >= len(vals):
                break
            lag = 24 * int(np.ceil(h / 24))
            th = hod_arr[j]
            rows.append((t0, h, j, th, hp.index[j].dayofweek, hp.index[j].month, last, m24,
                         vals[j - lag], np.median(hist[hist_hod == th]),
                         hw["temp"].values[j], hw["irr"].values[j], vals[j]))
    cols = ["issue", "h", "j", "hod", "dow", "month", "last", "m24", "same_h", "med7", "temp", "irr", "y"]
    return pd.DataFrame(rows, columns=cols)


FEATS = ["h", "hod", "dow", "month", "last", "m24", "same_h", "med7", "temp", "irr"]
issues23 = hp.index[(hp.index.year == 2023)]
issues24 = hp.index[(hp.index.year == 2024)]
P23 = price_design(issues23)
P24 = price_design(issues24)
price_model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                            l2_regularization=1.0, random_state=0)
price_model.fit(P23[FEATS], P23["y"].clip(-1.0, 1.0))  # winsorise extreme spikes for stable fit
P24["pred"] = price_model.predict(P24[FEATS])
P24["naive_med7"] = P24["med7"]
P24["persist"] = P24["last"]
pm = {}
for name in ["pred", "naive_med7", "persist"]:
    e = P24[name] - P24["y"]
    pm[name] = dict(MAE=float(e.abs().mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                    MAE_h1_6=float(e[P24.h <= 6].abs().mean()), MAE_h25_48=float(e[P24.h > 24].abs().mean()),
                    corr=float(np.corrcoef(P24[name], P24["y"])[0, 1]))
pd.DataFrame(pm).T.to_csv(RES / "price_forecast_metrics.csv")
log(pd.DataFrame(pm).T.round(4).to_string())
mae_by_h = P24.assign(ae=(P24["pred"] - P24["y"]).abs(), ae_n=(P24["med7"] - P24["y"]).abs()) \
    .groupby("h")[["ae", "ae_n"]].mean()
mae_by_h.to_csv(RES / "price_mae_by_horizon.csv")
price_fc_lookup = {t: g["pred"].values for t, g in P24.groupby("issue")}

# %% [markdown]
# ## 5. Battery model and optimisation (linear programme, HiGHS)
# Decision variables per 5-min step t: c_pv, c_grid (charge), d (discharge), gi, ge, cu (PV curtailment), soc.
#   Energy balance:  pv - cu - c_pv + d + gi = load + c_grid + ge
#   SoC dynamics:    soc[t+1] = soc[t] + (eta_pv c_pv + eta_g c_grid - d / eta_d) * dt
#   Limits:          soc_min E <= soc <= soc_max E ; c_pv + c_grid <= Pc(T) ; d <= Pd(T); c_pv <= pv - cu
#   Objective:       min sum (p_imp gi - p_exp ge + deg d) dt
# Curtailment is only useful when the export price is negative (wholesale case).

# %%
def power_limits(temp, P):
    f = np.ones_like(temp)
    hot = temp > A["derate_hot_start"]
    f[hot] = np.clip(1 - 0.05 * (temp[hot] - A["derate_hot_start"]), 0.5, 1)
    pc = P * f
    pd_ = P * f
    pc = np.where(temp < A["cold_charge_limit_c"], 0.5 * pc, pc)
    return pc, pd_


def bess_params(E=None, P=None, topo="AC", deg=None):
    E = A["bess_kwh"] if E is None else E
    P = A["bess_kw"] if P is None else P
    if topo == "AC":
        eff = (A["eta_c_pv"], A["eta_c_grid"], A["eta_d"])
    else:
        eff = (A["dc_eta_c_pv"], A["dc_eta_c_grid"], A["dc_eta_d"])
    return dict(E=E, P=P, eta_pv=eff[0], eta_g=eff[1], eta_d=eff[2],
                deg=A["deg_cost"] if deg is None else deg, topo=topo)


def solve_lp(load, pv, pimp, pexp, temp, bp, soc0, terminal_soc=None, allow_curtail=True,
             allow_grid_charge=True):
    T = len(load)
    E = bp["E"]
    I = sp.identity(T, format="csr")
    Z = sp.csr_matrix((T, T))
    Zs = sp.csr_matrix((T, T + 1))
    # variable order: c_pv, c_g, d, gi, ge, cu, soc(T+1)
    A1 = sp.hstack([-I, -I, I, I, -I, -I, Zs])
    b1 = load - pv
    D = sp.hstack([sp.csr_matrix((T, 1)), I]) - sp.hstack([I, sp.csr_matrix((T, 1))])
    A2 = sp.hstack([-bp["eta_pv"] * DT * I, -bp["eta_g"] * DT * I, DT / bp["eta_d"] * I, Z, Z, Z, D])
    n = 6 * T + T + 1
    rows = [A1, A2, sp.csr_matrix(([1.0], ([0], [6 * T])), shape=(1, n))]
    beq = [b1, np.zeros(T), [soc0]]
    if terminal_soc is not None:
        rows.append(sp.csr_matrix(([1.0], ([0], [n - 1])), shape=(1, n)))
        beq.append([terminal_soc])
    Aeq = sp.vstack(rows).tocsr()
    beq = np.concatenate([np.atleast_1d(x) for x in beq])
    # inequalities: c_pv + c_g <= pc ; c_pv + cu <= pv
    Aub = sp.vstack([sp.hstack([I, I, Z, Z, Z, Z, Zs]), sp.hstack([I, Z, Z, Z, Z, I, Zs])]).tocsr()
    pc, pdis = power_limits(temp, bp["P"])
    bub = np.r_[pc, pv]
    cost = np.r_[np.zeros(T), np.zeros(T), bp["deg"] * DT * np.ones(T), pimp * DT, -pexp * DT,
                 np.zeros(T), np.zeros(T + 1)]
    ub_cg = pc if allow_grid_charge else np.zeros(T)
    ub_cu = pv if allow_curtail else np.zeros(T)
    lb = np.zeros(n)
    ub = np.r_[pc, ub_cg, pdis, np.full(T, A["grid_kw"]), np.full(T, A["grid_kw"]), ub_cu,
               np.full(T + 1, bp["soc_max"] * E if "soc_max" in bp else A["soc_max"] * E)]
    lb[6 * T:] = A["soc_min"] * E
    r = linprog(cost, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=np.c_[lb, ub], method="highs")
    if r.status != 0:
        raise RuntimeError(r.message)
    x = r.x
    out = dict(c_pv=x[:T], c_g=x[T:2 * T], d=x[2 * T:3 * T], gi=x[3 * T:4 * T], ge=x[4 * T:5 * T],
               cu=x[5 * T:6 * T], soc=x[6 * T:])
    return out


def self_consumption(load, pv, temp, bp, soc0):
    """Default inverter mode: charge only from PV surplus, discharge to cover load, no grid charging."""
    T = len(load)
    pc, pdis = power_limits(temp, bp["P"])
    E = bp["E"]
    smin, smax = A["soc_min"] * E, A["soc_max"] * E
    c = np.zeros(T)
    d = np.zeros(T)
    soc = np.empty(T + 1)
    soc[0] = soc0
    s = soc0
    for t in range(T):
        net = load[t] - pv[t]
        if net < 0:
            c[t] = min(-net, pc[t], (smax - s) / (bp["eta_pv"] * DT))
            s += bp["eta_pv"] * c[t] * DT
        else:
            d[t] = min(net, pdis[t], (s - smin) * bp["eta_d"] / DT)
            s -= d[t] / bp["eta_d"] * DT
        soc[t + 1] = s
    g = load - pv + c - d
    return dict(c_pv=c, c_g=np.zeros(T), d=d, gi=np.clip(g, 0, None), ge=np.clip(-g, 0, None),
                cu=np.zeros(T), soc=soc)


def no_bess(load, pv, curtail_negative=None):
    T = len(load)
    g = load - pv
    ge = np.clip(-g, 0, None)
    cu = np.zeros(T)
    if curtail_negative is not None:
        cu = np.where(curtail_negative, ge, 0)
        ge = ge - cu
    return dict(c_pv=np.zeros(T), c_g=np.zeros(T), d=np.zeros(T), gi=np.clip(g, 0, None), ge=ge, cu=cu,
                soc=np.zeros(T + 1))


def evaluate(r, load, pv, pimp, pexp, months=12, vpp_fee=False, bp=None):
    gi, ge, d = r["gi"], r["ge"], r["d"]
    imp_cost = float((gi * pimp).sum() * DT)
    exp_rev = float((ge * pexp).sum() * DT)
    pv_used = pv - r["cu"]
    pv_to_grid = np.minimum(ge, pv_used)
    self_cons = float((pv_used.sum() - pv_to_grid.sum()) / pv.sum())
    out = dict(import_kWh=float(gi.sum() * DT), export_kWh=float(ge.sum() * DT),
               curtailed_kWh=float(r["cu"].sum() * DT), import_cost=imp_cost, export_revenue=exp_rev,
               energy_bill=imp_cost - exp_rev, vpp_fee=A["vpp_fee_month"] * months if vpp_fee else 0.0,
               self_consumption_pct=100 * self_cons,
               self_sufficiency_pct=100 * (1 - gi.sum() / load.sum()),
               discharge_kWh=float(d.sum() * DT),
               # true grid-sourced charging = charging while the site imports (the c_pv/c_grid split
               # is arbitrary for an AC-coupled battery because both have the same efficiency)
               grid_charge_kWh=float(np.minimum(r["c_pv"] + r["c_g"], gi).sum() * DT))
    out["annual_bill"] = out["energy_bill"] + out["vpp_fee"]
    if bp is not None and bp["E"] > 0:
        out["equiv_cycles"] = out["discharge_kWh"] / ((A["soc_max"] - A["soc_min"]) * bp["E"])
        out["wear_cost"] = out["discharge_kWh"] * bp["deg"]
    # energy balance closure
    bal = pv - r["cu"] - r["c_pv"] + r["d"] + gi - load - r["c_g"] - ge
    out["max_balance_error_kW"] = float(np.abs(bal).max())
    return out


# %% [markdown]
# ## 6. Site analysis (load vs PV) and perfect-foresight optimisation for each tariff

# %%
Y = {}
for yr in [2023, 2024]:
    m = clean.index.year == yr
    Y[yr] = {k: clean.loc[m, k].values for k in ["load", "pv", "temp", "imp_flat", "exp_flat", "imp_tou",
                                                  "exp_tou", "imp_ws", "exp_ws", "spot"]}
    Y[yr]["idx"] = clean.index[m]

site = {}
for yr in [2023, 2024]:
    ld, pv = Y[yr]["load"], Y[yr]["pv"]
    site[yr] = dict(load_MWh=ld.sum() * DT / 1000, pv_MWh=pv.sum() * DT / 1000,
                    load_kWh_day=ld.sum() * DT / len(ld) * STEPS_PER_DAY,
                    pv_kWh_day=pv.sum() * DT / len(pv) * STEPS_PER_DAY,
                    peak_load_kW=float(ld.max()), peak_pv_kW=float(pv.max()),
                    direct_self_cons_pct=100 * np.minimum(ld, pv).sum() / pv.sum())
# Energy that must come from somewhere when PV < load, per day ("evening/night deficit")
deficit = pd.Series(np.clip(clean["load"] - clean["pv"], 0, None) * DT, index=clean.index).resample("D").sum()
surplus = pd.Series(np.clip(clean["pv"] - clean["load"], 0, None) * DT, index=clean.index).resample("D").sum()
site["daily_deficit_kWh_pctiles"] = deficit.quantile([0.5, 0.8, 0.9, 0.95]).round(2).to_dict()
site["days_surplus_exceeds_deficit_pct"] = float(100 * (surplus > deficit).mean())
site["kWp_estimate"] = kwp_est
json.dump(site, open(RES / "site_summary.json", "w"), indent=2, default=float)
log(site)

TARIFFS = {"Flat": ("imp_flat", "exp_flat"), "ToU": ("imp_tou", "exp_tou"), "Wholesale": ("imp_ws", "exp_ws")}


def run_year(yr, bp, tariffs=TARIFFS, modes=("NoBESS", "SelfCons", "Optimised")):
    D = Y[yr]
    soc0 = 0.5 * bp["E"] if bp["E"] > 0 else 0
    res = {}
    sc = self_consumption(D["load"], D["pv"], D["temp"], bp, soc0) if bp["E"] > 0 else None
    for tn, (ci, ce) in tariffs.items():
        pi, pe = D[ci], D[ce]
        ws = tn == "Wholesale"
        if "NoBESS" in modes:
            res[(tn, "NoBESS")] = evaluate(no_bess(D["load"], D["pv"]), D["load"], D["pv"], pi, pe)
        if "SelfCons" in modes and sc is not None:
            res[(tn, "SelfCons")] = evaluate(sc, D["load"], D["pv"], pi, pe, bp=bp)
        if "Optimised" in modes and bp["E"] > 0:
            r = solve_lp(D["load"], D["pv"], pi, pe, D["temp"], bp, soc0, terminal_soc=soc0,
                         allow_curtail=ws)
            res[(tn, "Optimised")] = evaluate(r, D["load"], D["pv"], pi, pe, vpp_fee=ws, bp=bp)
            if ws:
                res[(tn, "Optimised")]["_r"] = r
    return res


if __name__ == "__main__":
    t0 = time.time()
    base = bess_params()
    pf = {}
    for yr in [2024, 2023]:
        pf[yr] = run_year(yr, base)
    rows = []
    for yr, rr in pf.items():
        for (tn, mode), v in rr.items():
            rows.append(dict(year=yr, tariff=tn, mode=mode, **{k: v[k] for k in v if not k.startswith("_")}))
    scen = pd.DataFrame(rows)
    nb = scen[scen["mode"] == "NoBESS"].set_index(["year", "tariff"])["annual_bill"]
    scb = scen[scen["mode"] == "SelfCons"].set_index(["year", "tariff"])["annual_bill"]
    scen["saving_vs_NoBESS"] = scen.apply(lambda r: nb[(r.year, r.tariff)] - r.annual_bill, axis=1)
    scen["saving_vs_SelfCons"] = scen.apply(lambda r: scb[(r.year, r.tariff)] - r.annual_bill, axis=1)
    scen.to_csv(RES / "scenarios_perfect_foresight.csv", index=False)
    log(scen[["year", "tariff", "mode", "annual_bill", "self_consumption_pct", "self_sufficiency_pct",
              "equiv_cycles", "saving_vs_NoBESS"]].round(2).to_string())
    check(scen["max_balance_error_kW"].max() < 1e-6, "energy balance closes in every scenario")

    # LP check: an LP that only minimises grid import reproduces the rule-based self-consumption
    D = Y[2024]
    r_lp = solve_lp(D["load"], D["pv"], np.ones(len(D["load"])), np.zeros(len(D["load"])), D["temp"],
                    dict(base, deg=0.0), 0.5 * base["E"], allow_curtail=False, allow_grid_charge=False)
    sc_lp_import = r_lp["gi"].sum() * DT
    sc_rule_import = pf[2024][("Flat", "SelfCons")]["import_kWh"]
    log(f"Self-consumption check: rule-based import {sc_rule_import:.1f} kWh vs LP minimum {sc_lp_import:.1f} kWh")
    check(sc_rule_import - sc_lp_import < 0.03 * sc_rule_import,
          "rule-based self-consumption within 3% of LP-optimal grid import")
    log(f"[{time.time() - t0:.0f}s] perfect foresight done")

# %% [markdown]
# ## 7. Battery sizing sweep and payback

# %%
def capex(E):
    return 0.0 if E == 0 else A["capex_fixed"] + A["capex_per_kwh"] * E


def npv_payback(saving, E, rebate=False):
    c = capex(E) - (A["rebate_per_kwh"] * E * (A["soc_max"] - A["soc_min"]) if rebate else 0)
    flows = [saving * (1 - A["fade_per_year"]) ** y for y in range(A["life_years"])]
    npv = -c + sum(f / (1 + A["discount_rate"]) ** (y + 1) for y, f in enumerate(flows))
    simple = c / saving if saving > 0 else np.inf
    return c, simple, npv


def size_job(E):
    P = 2.5 if E <= 2.5 else 5.0
    bp = bess_params(E=E, P=P)
    rr = run_year(2024, bp, tariffs={k: TARIFFS[k] for k in ["ToU", "Wholesale"]})
    return E, P, rr


if __name__ == "__main__":
    t0 = time.time()
    sizes = [2.5, 5.0, 7.5, 10.0, 13.5, 16.0, 20.0]
    out = pmap(size_job, sizes)
    nbt = pf[2024][("ToU", "NoBESS")]["annual_bill"]
    srows = []
    for E, P, rr in out:
        for (tn, mode), v in rr.items():
            if mode == "NoBESS":
                continue
            sav = (nbt if tn == "ToU" else nbt) - v["annual_bill"]  # vs no battery on ToU (best fixed tariff)
            c, pb, npv = npv_payback(sav, E)
            c_r, pb_r, npv_r = npv_payback(sav, E, rebate=True)
            srows.append(dict(kWh=E, kW=P, tariff=tn, mode=mode, annual_bill=v["annual_bill"],
                              saving_vs_ToU_noBESS=sav, self_consumption_pct=v["self_consumption_pct"],
                              self_sufficiency_pct=v["self_sufficiency_pct"], cycles=v.get("equiv_cycles"),
                              capex=c, payback_yr=pb, npv10=npv, payback_rebate_yr=pb_r, npv10_rebate=npv_r))
    sizing = pd.DataFrame(srows)
    sizing.to_csv(RES / "sizing_sweep.csv", index=False)
    log(sizing.round(2).to_string())
    log(f"[{time.time() - t0:.0f}s] sizing done")

# %% [markdown]
# ## 8. 48-hour rolling-horizon scheduler (hourly re-plan, 5-minute execution)
# At the top of every hour: forecast PV (model on weather) and price (section 4) for 48 h,
# solve the LP with the actual SoC, then execute the first hour of battery set-points.
# Each 5-minute step the battery follows its set-point without knowing that interval's price or PV;
# the grid absorbs the PV/load mismatch; export is curtailed when the forecast export price is negative.
# Bills are settled with actual 5-minute prices and actual PV (commercial series, per the brief).

# %%
def rolling(args):
    yr, tariff, pv_source, price_source, bp = args[:5]
    rt = len(args) > 5 and args[5]  # real-time override using the last settled 5-minute price
    D = Y[yr]
    ci, ce = TARIFFS[tariff]
    load, pv_act, temp = D["load"], D["pv"], D["temp"]
    pimp_act, pexp_act = D[ci], D[ce]
    idx = D["idx"]
    m = clean.index.year == yr
    pv_fc_all = clean.loc[m, pv_source].values if pv_source != "actual" else pv_act
    adders_y = pimp_act - D["spot"] if tariff == "Wholesale" else None
    T = len(load)
    soc = 0.5 * bp["E"]
    rec = {k: np.zeros(T) for k in ["c_pv", "c_g", "d", "gi", "ge", "cu"]}
    rec["soc"] = np.zeros(T + 1)
    rec["soc"][0] = soc
    for k0 in range(0, T, STEPS_PER_HOUR):
        k1 = min(T, k0 + HORIZON)
        sl = slice(k0, k1)
        n = k1 - k0
        if tariff == "Wholesale" and price_source == "forecast":
            fc = price_fc_lookup.get(idx[k0])
            spot_fc = np.repeat(fc, STEPS_PER_HOUR)[:n]
            if len(spot_fc) < n:
                spot_fc = np.r_[spot_fc, np.full(n - len(spot_fc), spot_fc[-1])]
            pi_f, pe_f = spot_fc + adders_y[sl], spot_fc
        else:
            pi_f, pe_f = pimp_act[sl], pexp_act[sl]
        r = solve_lp(load[sl], pv_fc_all[sl], pi_f, pe_f, temp[sl], bp, soc,
                     allow_curtail=(tariff == "Wholesale"))
        e = min(STEPS_PER_HOUR, n)
        for j in range(e):
            t = k0 + j
            c_pv, c_g, d = r["c_pv"][j], r["c_g"][j], r["d"][j]
            curtail = pe_f[j] < 0 and tariff == "Wholesale"
            if rt and t > 0:
                p_last = D["spot"][t - 1]  # known: settled at the end of the previous interval
                pc_t, pd_t = power_limits(temp[t:t + 1], bp["P"])
                if p_last >= A["rt_spike"]:      # spike in progress -> discharge at full power
                    c_pv, c_g, d = 0.0, 0.0, float(pd_t[0])
                    curtail = False
                elif p_last < 0:                 # negative price in progress -> stop exporting, soak up PV
                    curtail = True
                    d = 0.0
                    c_pv = float(min(pc_t[0], max(pv_fc_all[t] - load[t], 0.0)))
                    c_g = 0.0
            # SoC safety clip with actual state
            smax, smin = A["soc_max"] * bp["E"], A["soc_min"] * bp["E"]
            ch = c_pv * bp["eta_pv"] + c_g * bp["eta_g"]
            if ch > 0 and soc + ch * DT > smax + 1e-9:
                f = (smax - soc) / (ch * DT)
                c_pv, c_g = c_pv * f, c_g * f
            if d > 0 and soc - d / bp["eta_d"] * DT < smin - 1e-9:
                d = max(0.0, (soc - smin) * bp["eta_d"] / DT)
            # charging labelled "from PV" can only use actual PV; any shortfall comes from the grid
            c_tot = c_pv + c_g
            c_pv_a = min(c_pv, pv_act[t])
            c_g_a = c_tot - c_pv_a
            soc += (c_pv_a * bp["eta_pv"] + c_g_a * bp["eta_g"] - d / bp["eta_d"]) * DT
            netload = load[t] + c_tot - d - pv_act[t]
            gi = max(netload, 0.0)
            surplus = max(-netload, 0.0)
            rec["c_pv"][t], rec["c_g"][t], rec["d"][t] = c_pv_a, c_g_a, d
            rec["gi"][t] = gi
            rec["cu"][t] = surplus if curtail else 0.0
            rec["ge"][t] = 0.0 if curtail else surplus
            rec["soc"][t + 1] = soc
    ev = evaluate(rec, load, pv_act, pimp_act, pexp_act, vpp_fee=(tariff == "Wholesale"), bp=bp)
    # value the terminal SoC difference at the mean import price so runs are comparable
    ev["terminal_soc_kWh"] = float(soc)
    ev["annual_bill_adj"] = ev["annual_bill"] - (soc - 0.5 * bp["E"]) * bp["eta_d"] * float(np.mean(pimp_act))
    return (yr, tariff, pv_source, price_source + ("+RT" if rt else "")), ev, rec


if __name__ == "__main__":
    t0 = time.time()
    jobs = [
        (2024, "Wholesale", "actual", "actual", base),     # rolling, perfect forecasts (horizon effect only)
        (2024, "Wholesale", PV_FC, "actual", base),        # PV forecast error only
        (2024, "Wholesale", "actual", "forecast", base),   # price forecast error only
        (2024, "Wholesale", PV_FC, "forecast", base),      # realistic VPP-SPREE scheduler
        (2024, "Wholesale", "pv_persist", "forecast", base),  # stress: weather-free PV forecast
        (2024, "Wholesale", PV_FC, "forecast", base, True),   # + 5-minute real-time override (refined VPP)
        (2024, "ToU", PV_FC, "actual", base),              # ToU with PV forecast
        (2024, "ToU", "actual", "actual", base),
        (2024, "Flat", PV_FC, "actual", base),
    ]
    rolled = pmap(rolling, jobs)
    rrows = []
    REC = {}
    for key, ev, rec in rolled:
        rrows.append(dict(year=key[0], tariff=key[1], pv=key[2], price=key[3], **ev))
        REC[key] = rec
    rolled_dict = REC
    roll = pd.DataFrame(rrows)
    roll.to_csv(RES / "scenarios_rolling_horizon.csv", index=False)
    log(roll[["tariff", "pv", "price", "annual_bill", "annual_bill_adj", "self_consumption_pct",
              "equiv_cycles", "terminal_soc_kWh"]].round(2).to_string())
    log(f"[{time.time() - t0:.0f}s] rolling horizon done")
    check(all(r["soc"].min() >= A["soc_min"] * base["E"] - 1e-6 and r["soc"].max() <= A["soc_max"] * base["E"] + 1e-6
              for r in REC.values()), "rolling-horizon SoC stays within limits")
    check(roll["max_balance_error_kW"].max() < 1e-6, "rolling-horizon energy balance closes")

# %% [markdown]
# ## 9. Sensitivities: degradation cost and topology (perfect foresight, wholesale & ToU, 2024)

# %%
def sens_job(args):
    label, bp = args
    rr = run_year(2024, bp, tariffs={k: TARIFFS[k] for k in ["ToU", "Wholesale"]}, modes=("Optimised", "SelfCons"))
    return label, bp, rr


if __name__ == "__main__":
    t0 = time.time()
    sj = [(f"deg={d}", bess_params(deg=d)) for d in [0.0, 0.02, 0.05, 0.10, 0.20]]
    sj += [("DC-coupled", bess_params(topo="DC"))]
    so = pmap(sens_job, sj)
    srows = []
    for label, bp, rr in so:
        for (tn, mode), v in rr.items():
            srows.append(dict(case=label, deg=bp["deg"], topo=bp["topo"], tariff=tn, mode=mode,
                              annual_bill=v["annual_bill"], cycles=v.get("equiv_cycles"),
                              wear_cost=v.get("wear_cost"), grid_charge_kWh=v["grid_charge_kWh"],
                              self_consumption_pct=v["self_consumption_pct"]))
    sens = pd.DataFrame(srows)
    sens["bill_plus_wear"] = sens["annual_bill"] + sens["wear_cost"].fillna(0)
    sens.to_csv(RES / "sensitivity_deg_topology.csv", index=False)
    log(sens.round(2).to_string())
    log(f"[{time.time() - t0:.0f}s] sensitivities done")

# %% [markdown]
# ## 10. VPP product techno-economics and summary

# %%
if __name__ == "__main__":
    S = scen[scen.year == 2024].set_index(["tariff", "mode"])
    R = roll.set_index(["tariff", "pv", "price"])
    best_fixed = S.loc[[("Flat", "SelfCons"), ("ToU", "SelfCons"), ("ToU", "Optimised")], "annual_bill"]
    vpp_real = R.loc[("Wholesale", PV_FC, "forecast"), "annual_bill_adj"]
    vpp_pf = S.loc[("Wholesale", "Optimised"), "annual_bill"]
    vpp_rt = R.loc[("Wholesale", PV_FC, "forecast+RT"), "annual_bill_adj"]
    summary = dict(
        price_source=PRICE_SOURCE, pv_forecast_model=PV_FC, kWp_estimate=kwp_est,
        site=site, missing=miss, spurious_flags=flags, pv_metrics=pv_metrics, price_metrics=pm,
        bills_2024={" | ".join(k): v for k, v in S["annual_bill"].round(2).to_dict().items()},
        rolling_2024={" | ".join(k): v for k, v in R["annual_bill_adj"].round(2).to_dict().items()},
        best_fixed_tariff_bill=float(best_fixed.min()), best_fixed_tariff_case=str(best_fixed.idxmin()),
        vpp_realistic_bill=float(vpp_real), vpp_perfect_bill=float(vpp_pf),
        vpp_refined_rt_bill=float(vpp_rt),
        vpp_refined_benefit_vs_best_fixed=float(best_fixed.min() - vpp_rt),
        vpp_net_customer_benefit_vs_best_fixed=float(best_fixed.min() - vpp_real),
        vpp_fee_year=12 * A["vpp_fee_month"],
        forecast_value_loss_total=float(vpp_real - vpp_pf),
        loss_pv_only=float(R.loc[("Wholesale", PV_FC, "actual"), "annual_bill_adj"] - R.loc[("Wholesale", "actual", "actual"), "annual_bill_adj"]),
        loss_price_only=float(R.loc[("Wholesale", "actual", "forecast"), "annual_bill_adj"] - R.loc[("Wholesale", "actual", "actual"), "annual_bill_adj"]),
        loss_horizon=float(R.loc[("Wholesale", "actual", "actual"), "annual_bill_adj"] - vpp_pf),
    )
    # Where is value lost? Split net grid cash flow by actual spot-price band
    D = Y[2024]
    bands = [(-9, 0, "negative"), (0, 0.1, "0-100 $/MWh"), (0.1, 0.3, "100-300"), (0.3, 99, ">=300 (spikes)")]
    rr_pf = pf[2024][("Wholesale", "Optimised")]["_r"]
    rr_rl = rolled_dict[(2024, "Wholesale", PV_FC, "forecast")]
    brow = []
    for lo, hi, nm in bands:
        m = (D["spot"] >= lo) & (D["spot"] < hi)
        for lab, rr in [("perfect", rr_pf), ("forecast", rr_rl)]:
            cash = float(((rr["ge"] * D["exp_ws"] - rr["gi"] * D["imp_ws"]) * DT)[m].sum())
            brow.append(dict(band=nm, run=lab, share_of_intervals=float(m.mean()), net_credit=cash,
                             discharge_kWh=float((rr["d"] * DT)[m].sum())))
    bandtab = pd.DataFrame(brow)
    bandtab.to_csv(RES / "value_by_price_band.csv", index=False)
    log(bandtab.round(2).to_string())
    json.dump(summary, open(RES / "summary.json", "w"), indent=2, default=str)
    log(json.dumps({k: v for k, v in summary.items() if k not in ("site", "missing", "spurious_flags", "pv_metrics", "price_metrics")}, indent=1, default=str))

# %% [markdown]
# ## 11. Figures

# %%
if __name__ == "__main__":
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})
    C = {"load": "#2E5A88", "pv": "#E8A33D", "batt": "#3C9D5D", "grid": "#7F7F7F", "price": "#B03A48"}

    # Fig 1 typical days by season (load vs PV) + daily energy
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    seasons = {"Summer (DJF)": [12, 1, 2], "Autumn/Spring": [3, 4, 5, 9, 10, 11], "Winter (JJA)": [6, 7, 8]}
    x = clean.index.hour + clean.index.minute / 60
    for i, (nm, ms) in enumerate(seasons.items()):
        m = clean.index.month.isin(ms)
        g = clean.loc[m, ["load", "pv"]].groupby(x[m]).mean()
        ax[i].fill_between(g.index, g["pv"], color=C["pv"], alpha=0.35, label="PV")
        ax[i].plot(g.index, g["load"], color=C["load"], lw=2, label="Load")
        ax[i].set_title(nm); ax[i].set_xlabel("Hour of day (AEST)"); ax[i].set_xticks([0, 6, 12, 18, 24])
    ax[0].set_ylabel("Mean power (kW)"); ax[0].legend()
    fig.suptitle("Fig 1. Average daily load and PV profiles, 2023-2024: PV is ~3.5x load, deficits are evening/night")
    fig.tight_layout(); fig.savefig(RES / "fig01_typical_days.png"); plt.close(fig)

    # Fig 2 data quality
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    dmiss = df["load"].isna().resample("D").sum() / STEPS_PER_DAY * 100
    ax[0].bar(dmiss.index, dmiss.values, width=1.5, color=C["price"])
    ax[0].set_ylabel("% of day missing"); ax[0].set_title("Missing load/weather intervals per day")
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    s = clean.loc[train_mask]
    ax[1].scatter(s["irr"][::5], s["pv"][::5] if "pv" in s else tr["pv_meas"][::5], s=1, alpha=0.2, color=C["pv"], label="2023 measured")
    gg = np.linspace(0, 1100, 50)
    ax[1].plot(gg, k_fit * gg / 1000, color="k", lw=1, label=f"Fitted {k_fit:.1f} kW per kW/m²")
    ax[1].set_xlabel("Irradiance (W/m²)"); ax[1].set_ylabel("PV output (kW)"); ax[1].set_title("PV vs irradiance (2023)")
    ax[1].legend(markerscale=8)
    fig.suptitle("Fig 2. Data quality: 30 logger outages (max one full day, 30 Nov 2024); PV consistent with irradiance")
    fig.tight_layout(); fig.savefig(RES / "fig02_data_quality.png"); plt.close(fig)

    # Fig 3 PV forecast vs commercial (example week + metrics)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [2.2, 1]})
    wk = clean.loc["2024-10-07":"2024-10-13"]
    ax[0].plot(wk.index, wk["pv"], color="k", lw=1.2, label="Commercial (treated as actual)")
    ax[0].plot(wk.index, wk[PV_FC], color=C["pv"], lw=1, label="Our model (2024 weather)")
    ax[0].plot(wk.index, wk["pv_persist"], color=C["grid"], lw=0.8, ls="--", label="Persistence (yesterday)")
    ax[0].set_ylabel("kW"); ax[0].legend(fontsize=7); ax[0].set_title("Example week, Oct 2024")
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    names = ["pv_phys", "pv_ml", "pv_persist"]
    labels = ["Physical PR", "ML (GBM)", "Persistence"]
    vals = [pv_metrics[n + "_test_2024_vs_commercial"]["nRMSE_cap_pct"] for n in names]
    ax[1].bar(labels, vals, color=[C["grid"], C["pv"], "#BBBBBB"])
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.2, f"{v:.1f}%", ha="center")
    ax[1].set_ylabel("nRMSE (% of kWp)"); ax[1].set_title("2024 test error")
    fig.suptitle("Fig 3. PV model trained on 2023, tested on 2024 against the commercial forecast")
    fig.tight_layout(); fig.savefig(RES / "fig03_pv_forecast.png"); plt.close(fig)

    # Fig 4 price forecast
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [2.2, 1]})
    ex = P24[P24.issue == pd.Timestamp("2024-06-11 07:00")]
    if len(ex) == 0:
        ex = P24[P24.issue == P24.issue.iloc[len(P24) // 2]]
    tt = hp.index[ex["j"].values]
    ax[0].step(tt, ex["y"] * 1000, where="post", color="k", label="Actual hourly price")
    ax[0].step(tt, ex["pred"] * 1000, where="post", color=C["price"], label="48 h forecast")
    ax[0].step(tt, ex["med7"] * 1000, where="post", color=C["grid"], ls="--", label="7-day median profile")
    ax[0].set_ylabel("$/MWh"); ax[0].legend(fontsize=7); ax[0].set_title(f"Forecast issued {ex.issue.iloc[0]:%d %b %Y %H:%M}")
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
    ax[1].plot(mae_by_h.index, mae_by_h["ae"] * 1000, color=C["price"], label="GBM model")
    ax[1].plot(mae_by_h.index, mae_by_h["ae_n"] * 1000, color=C["grid"], ls="--", label="7-day median")
    ax[1].set_xlabel("Horizon (h)"); ax[1].set_ylabel("MAE ($/MWh)"); ax[1].legend(fontsize=7)
    ax[1].set_title("2024 error vs horizon")
    fig.suptitle("Fig 4. Wholesale price forecast (" + ("placeholder 2024 prices" if "PLACEHOLDER" in PRICE_SOURCE else "AEMO 2024") + ")")
    fig.tight_layout(); fig.savefig(RES / "fig04_price_forecast.png"); plt.close(fig)

    # Fig 5 sizing
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for (tn, mode), g in sizing.groupby(["tariff", "mode"]):
        ax[0].plot(g["kWh"], g["saving_vs_ToU_noBESS"], marker="o", label=f"{tn} - {mode}")
        ax[1].plot(g["kWh"], g["payback_yr"].clip(upper=40), marker="o", label=f"{tn} - {mode}")
    ax[0].set_xlabel("Battery size (kWh)"); ax[0].set_ylabel("Annual saving ($) vs no battery on ToU")
    ax[1].set_xlabel("Battery size (kWh)"); ax[1].set_ylabel("Simple payback (years, capped 40)")
    ax[1].axhline(10, color="k", lw=0.8, ls=":"); ax[0].legend(fontsize=7)
    fig.suptitle("Fig 5. Battery sizing (2024): savings flatten beyond ~7.5-10 kWh because the night deficit is small")
    fig.tight_layout(); fig.savefig(RES / "fig05_sizing.png"); plt.close(fig)

    # Fig 6 annual bills
    fig, ax = plt.subplots(figsize=(8, 3.4))
    order = [("Flat", "NoBESS"), ("Flat", "SelfCons"), ("Flat", "Optimised"), ("ToU", "NoBESS"), ("ToU", "SelfCons"),
             ("ToU", "Optimised"), ("Wholesale", "NoBESS"), ("Wholesale", "SelfCons"), ("Wholesale", "Optimised")]
    vals = [S.loc[o, "annual_bill"] for o in order] + [vpp_real, vpp_rt]
    labs = [f"{a}\n{b}" for a, b in order] + ["Wholesale\nVPP forecast", "Wholesale\nVPP fc + RT"]
    cols = [C["grid"] if "NoBESS" in l else (C["batt"] if "SelfCons" in l else C["load"]) for l in labs]
    cols[-1] = cols[-2] = C["price"]
    ax.bar(range(len(vals)), vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + (8 if v >= 0 else -8), f"${v:,.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    ax.set_ylim(min(vals) * 1.15 - 20, max(vals) * 1.15 + 20)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel("Annual energy bill 2024 ($)")
    ax.set_title("Fig 6. Annual bill by tariff and battery strategy (10 kWh / 5 kW; wholesale incl. $300 VPP fee)")
    fig.tight_layout(); fig.savefig(RES / "fig06_bills.png"); plt.close(fig)

    # Fig 7 dispatch example: wholesale VPP rolling vs ToU optimised, same 3 days
    key = (2024, "Wholesale", PV_FC, "forecast")
    rec = REC[key]
    i0 = int(np.where(Y[2024]["idx"] == pd.Timestamp("2024-06-10"))[0][0])
    sl = slice(i0, i0 + 3 * STEPS_PER_DAY)
    tt = Y[2024]["idx"][sl]
    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(tt, Y[2024]["load"][sl], color=C["load"], label="Load")
    ax[0].fill_between(tt, Y[2024]["pv"][sl], color=C["pv"], alpha=0.4, label="PV")
    ax[0].plot(tt, rec["d"][sl] - rec["c_pv"][sl] - rec["c_g"][sl], color=C["batt"], label="Battery (+discharge)")
    ax[0].set_ylabel("kW"); ax[0].legend(fontsize=7, ncol=3)
    ax[1].plot(tt, rec["soc"][sl] / base["E"] * 100, color=C["batt"])
    ax[1].set_ylabel("SoC (%)")
    ax[2].plot(tt, Y[2024]["spot"][sl] * 1000, color=C["price"], lw=0.8, label="Actual spot")
    fcs = np.full(3 * STEPS_PER_DAY, np.nan)
    ax[2].set_ylabel("$/MWh"); ax[2].legend(fontsize=7)
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    fig.suptitle("Fig 7. VPP scheduler (48 h rolling, forecasts) - three June 2024 days")
    fig.tight_layout(); fig.savefig(RES / "fig07_dispatch_example.png"); plt.close(fig)

    # Fig 8 value of information waterfall
    steps = [("Perfect foresight\n(full-year LP)", -vpp_pf),
             ("48 h horizon", -summary["loss_horizon"]),
             ("PV forecast\nerror", -summary["loss_pv_only"]),
             ("Price forecast\nerror", -summary["loss_price_only"])]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    cum = steps[0][1]
    ax.bar(0, cum, color=C["load"]); ax.text(0, cum + 5, f"${cum:,.0f}", ha="center", fontsize=8)
    for i, (nm, v) in enumerate(steps[1:], start=1):
        ax.bar(i, v, bottom=cum, color=C["price"])
        ax.text(i, cum + 5, f"-${-v:,.0f}", ha="center", fontsize=8)
        cum += v
    for j, (nm, v, col) in enumerate([("Realistic VPP\n(both forecasts)", -vpp_real, C["grid"]),
                                       ("+ 5-min real-time\noverride", -vpp_rt, C["batt"])], start=len(steps)):
        ax.bar(j, v, color=col); ax.text(j, v + 5, f"${v:,.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(steps) + 2))
    ax.set_xticklabels([s_[0] for s_ in steps] + ["Realistic VPP\n(both forecasts)", "+ 5-min real-time\noverride"], fontsize=7)
    ax.set_ylabel("Annual net credit 2024 ($, incl. VPP fee)")
    ax.set_ylim(min(0, -vpp_real) - 20, max(-vpp_pf, -vpp_rt) * 1.15 + 10)
    ax.set_title("Fig 8. Value lost to uncertainty (each effect measured on its own): price error dominates")
    fig.tight_layout(); fig.savefig(RES / "fig08_uncertainty.png"); plt.close(fig)

    # Fig 9 degradation sensitivity
    g = sens[(sens["case"].str.startswith("deg")) & (sens["mode"] == "Optimised")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for tn, gg in g.groupby("tariff"):
        ax[0].plot(gg["deg"], gg["cycles"], marker="o", label=tn)
        ax[1].plot(gg["deg"], gg["annual_bill"], marker="o", label=tn)
    ax[0].set_xlabel("Degradation cost ($/kWh discharged)"); ax[0].set_ylabel("Equivalent full cycles / year")
    ax[1].set_xlabel("Degradation cost ($/kWh discharged)"); ax[1].set_ylabel("Annual bill ($, excl. wear)")
    ax[0].legend()
    fig.suptitle("Fig 9. Degradation cost in the objective controls cycling with little bill penalty")
    fig.tight_layout(); fig.savefig(RES / "fig09_degradation.png"); plt.close(fig)

    # Fig 10 monthly bill: ToU self-cons vs VPP realistic
    mi = Y[2024]["idx"].month
    tou_sc = self_consumption(Y[2024]["load"], Y[2024]["pv"], Y[2024]["temp"], base, 0.5 * base["E"])
    mb_tou = pd.Series((tou_sc["gi"] * Y[2024]["imp_tou"] - tou_sc["ge"] * Y[2024]["exp_tou"]) * DT).groupby(mi).sum()
    mb_vpp = pd.Series((rec["gi"] * Y[2024]["imp_ws"] - rec["ge"] * Y[2024]["exp_ws"]) * DT).groupby(mi).sum() + A["vpp_fee_month"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    w = 0.4
    ax.bar(mb_tou.index - w / 2, mb_tou.values, w, color=C["batt"], label="ToU + self-consumption battery")
    ax.bar(mb_vpp.index + w / 2, mb_vpp.values, w, color=C["price"], label="Wholesale + VPP-SPREE (incl. $25 fee)")
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month 2024"); ax.set_ylabel("Monthly bill ($)"); ax.legend(fontsize=7)
    ax.set_title("Fig 10. Monthly bills: where the VPP adds (or loses) value")
    fig.tight_layout(); fig.savefig(RES / "fig10_monthly.png"); plt.close(fig)
    pd.DataFrame({"tou_selfcons": mb_tou, "vpp_wholesale": mb_vpp}).to_csv(RES / "monthly_bills_2024.csv")

    # interval audit for the realistic VPP run
    aud = pd.DataFrame({k: v[:-1] if k == "soc" else v for k, v in rec.items()}, index=Y[2024]["idx"])
    aud["load"], aud["pv"], aud["spot"] = Y[2024]["load"], Y[2024]["pv"], Y[2024]["spot"]
    aud.round(4).to_csv(RES / "interval_vpp_2024.csv.gz", compression="gzip")
    (RES / "validation_log.txt").write_text("\n".join(LOG))
    log("done")
