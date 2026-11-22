# SOLA4237 Assignment 2: household BESS optimisation (Shepparton, VIC)

Reproducible analysis package for the VPP-SPREE single-site BESS optimisation task.

## Run

```bash
python -m pip install -r requirements.txt
python analysis.py          # ~15-20 min on 4 cores; set N_JOBS=1 to run serially
```

`analysis.ipynb` contains exactly the same code split into cells (Run All from this folder).
All outputs are written to `results/`.

## Data

- `data/Data_VIC_Shepparton - Hot Dry Summer, Cool Winter_2023-2024_partial.csv`: supplied, unchanged.
  2023 = year 1 (measured PV, spot price); 2024 = year 2 (commercial PV forecast, no spot price).
- `data/aemo/PRICE_AND_DEMAND_2024MM_VIC1.csv` (+ `202501`): official AEMO VIC1 5-minute prices for 2024,
  joined exactly on `SETTLEMENTDATE`. **If these files are missing the script uses a clearly-labelled
  placeholder** (2023 prices shifted 52 weeks); `results/summary.json -> price_source` states which was used.

## Method in one paragraph

Telemetry is checked (grid regularity, DST, gaps, physical bounds, flatlines) and gaps are filled (up to 12 missing values are linearly interpolated, including the leading part of longer gaps; remaining values use a centred seven-day same-slot mean with fallback filling). A PV model (physical PR model and gradient-boosted
trees on irradiance/temperature/time) is trained on 2023 and tested on 2024 against the commercial forecast.
A price model (gradient-boosted trees) issues 48 hourly price forecasts every hour. The battery is a linear
programme (HiGHS via `scipy.optimize.linprog`) with SoC limits, charge/discharge efficiencies (AC- or
DC-coupled), temperature derating, a degradation cost per kWh discharged and PV curtailment at negative
prices. It is solved with perfect foresight for Flat, ToU and wholesale passthrough tariffs, then inside a
48-hour rolling-horizon scheduler (re-planned hourly, executed every 5 minutes without knowing the current
price/PV, settled at actual values). Sizing, degradation cost and topology sensitivities and payback are reported.

## Key outputs (`results/`)

| File | Content |
|---|---|
| `summary.json` | all headline numbers used in the report |
| `assumptions.json` | every assumed parameter |
| `scenarios_perfect_foresight.csv` | bills/self-consumption for 3 tariffs x {no BESS, self-consumption, optimised}, 2023 and 2024 |
| `scenarios_rolling_horizon.csv` | rolling-horizon runs (perfect vs forecast PV/price, real-time override) |
| `sizing_sweep.csv` | 2.5-20 kWh, savings, payback, NPV, with/without rebate |
| `sensitivity_deg_topology.csv` | degradation cost 0-0.20 $/kWh and DC-coupled case |
| `pv_model_metrics.csv`, `price_forecast_metrics.csv`, `price_mae_by_horizon.csv` | forecast accuracy |
| `value_by_price_band.csv` | where the VPP earns/loses value (by spot price band) |
| `monthly_bills_2024.csv`, `interval_vpp_2024.csv.gz` | monthly bills and 5-minute audit of the VPP run |
| `validation_log.txt` | automated checks that passed in the executed run |
| `fig01`-`fig10` | report figures |

## Conventions

Powers in kW (AC), energy = kW x 5/60 h. Positive battery = discharge. Prices AUD/kWh excl. GST.
Bills are energy-only (no daily supply charge, identical across options) plus the $25/month VPP fee for
the wholesale product. Timestamps are AEST (NEM time) without daylight saving.

## Report revision and billing reconciliation

The submitted report uses a consistent AUD 300 annual subscription fee for every wholesale scenario.
The saved analysis assigns zero service fee to the wholesale `NoBESS` and `SelfCons` baselines,
while wholesale `Optimised` and rolling-horizon cases already include it. To reconcile the report,
add AUD 300 to those two baseline bills in each year. This changes accounting only, not dispatch.
See `report_revision/bill_reconciliation.csv`; the original results remain unchanged.

The report numbers for controlled wholesale, ToU, sizing, forecast errors and payback are unchanged.
The report's figure numbering differs from the original `fig01`–`fig10` filenames.

## Verification and AI assistance

The author confirmed rerunning the analysis locally on 22 November 2026.
During report revision, Codex independently re-settled the saved five-minute forecast-only
wholesale audit, checked energy balances and SoC limits, and cross-checked CSV bill identities.
The rounded interval audit reproduces the unadjusted bill within AUD 0.02 of the full-precision
summary. This check was not a fresh optimisation run.

`AI_USE_LOG.md` records Claude's analysis assistance and the later author-confirmed Codex
prose drafting and revision. The author supplied the report outline and ideas.

## Access for assessment

This package is prepared for a private repository. The marker needs GitHub access to a private
repository; the URL alone does not grant access. Retain the supplied course data privately.
