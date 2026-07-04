# German Electricty Price

**Live dashboard:** https://andreacermaria.github.io/german-power-forecasting/

A day-ahead price forecasting and risk infrastructure project for the German-Luxembourg (DE-LU) power bidding zone.

## Project

Day-ahead power prices are fat tailed: most hours are calm, but a small number spike violently when renewable output drops and demand rises. This project produces **one predictive distribution, read two ways**:

- **As a price forecast** — a median estimate 
- **As a risk measure** — the capital that would be needed to hold against uncertainty

## Data

**Price and fundamentals — ENTSO-E Transparency Platform**
Hourly day-ahead auction price, load forecast, and wind/solar generation forecast for the DE-LU bidding zone, from 2019-01-01 onward. Pulled via the ENTSO-E API using a token loaded from `.env`.

**Outage events — EEX Group Transparency API (REMIT/UMM)**
A full 5-year backfill of unplanned outage messages (987,439 events) scraped from `api.eex-group.com`, scoped with `country=DE`.

All series are merged into a single hourly panel: `de_lu_features.parquet` — 19 columns, timezone-aware (`Europe/Brussels`), zero nulls, starting 2019-01-01.

## Model

**HistGradientBoostingRegressor** (scikit-learn), trained as a **quantile regression ensemble** producing five quantiles (q05, q25, q50, q75, q95) per hour, wrapped in **conformalized quantile regression (CQR)** for calibrated prediction intervals.

Validation is a **walk forward backtest over 49 monthly blocks** — the model is retrained at each step using only data available up to that point in time, so the reported accuracy reflects what a trader or risk desk would actually have seen in production, not a model trained with hindsight.

### Key variables

| Feature | Role |
|---|---|
| `residual_load` (load minus renewable forecast) | Dominant driver |
| `price_roll168_mean` / `price_roll168_std` | Backward looking price level and volatility |
| `unplanned_mw_offline_dispatchable` | Unplanned dispatchable capacity offline, from UMM events, weak signal |
| `spike_freq_168` | Rolling 168h fraction of hours exceeding `price_roll168_mean + 2·price_roll168_std`, shifted 24h, added to give the model better information about the current spike regime |
| Wind/solar forecast components | Near zero importance due to collinearity with `residual_load` |

**No future information ever enters a forecast.** Every rolling or lagged feature is shifted a minimum of 24 hours.

### Calibration

An early calibration regression (q50 empirical coverage 0.39 vs. nominal 0.50) was traced to a training-window bug, not the CQR method itself. A separate regime-transition issue in 2021–2022 — where price rose faster than the backward-looking `price_roll168_mean` anchor could track — was fixed by replacing symmetric CQR with a **per-quantile shift**, computed independently for each of the five quantiles rather than one shared correction. This improved coverage across all quantiles (e.g. q50 empirical coverage: 0.391 → 0.467) and cut spike-hour MAE from 91.0 to 81.8.

Hand-tuned tail-weighting multipliers were explicitly rejected: they're not defensible in a risk-modeling context, since there's no principled way to justify the tuning after the fact. The residual upper-tail (q95) gap is instead being addressed through better model information — `spike_freq_168` — rather than post-hoc statistical correction.

## Risk module

Built on top of the same quantile forecasts:

- **Value-at-Risk (VaR) and Expected Shortfall (ES)** at the 99% confidence level, derived from the model's predictive distribution
- **Initial margin**, which extends the VaR horizon to cover a realistic close out period for a position
- **Procyclicality**, tracked by recomputing the margin figure daily over a rolling one-year lookback window — this is what makes a past price spike visible in the collateral requirement for months afterward, until it ages out of the window

## The dashboard

The dashboard is a **static, client side page**, all figures are pre computed from the backtest.


