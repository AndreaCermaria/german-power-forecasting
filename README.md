# German Electricity Price

**Live dashboard:** https://andreacermaria.github.io/german-power-forecasting/

A day ahead price forecasting and risk infrastructure project for the German Luxembourg (DE-LU) power bidding zone.

## Project

Day ahead power prices are fat tailed: most hours are calm, but a small number spike violently when renewable output drops and demand rises. This project produces **one predictive distribution, read two ways**:

- **As a price forecast**, a median estimate, with a full band of uncertainty around it
- **As a risk measure**, the capital that would be needed to hold against that uncertainty

## How it's built

The project moves through four stages, each building on the last:

1. **Forecasting engine**: pull five years of German market fundamentals, and train a model that outputs a full predictive distribution (five quantiles) for the next day ahead price, retrained at every step of a walk forward backtest, never trained with hindsight (considering the actual price the day after).
2. **Risk layer**: calculation of Value at Risk, Expected Shortfall, and the initial margin a position would require, 
3. **Feature testing**: test whether adding unplanned power plant outage data sharpens the model's read on price spikes specifically.
4. **Delivery**: package the forecast and the risk numbers into a single interactive dashboard.

## Data

**Price and fundamentals, ENTSO-E Transparency Platform.** Hourly day ahead auction price, load forecast, and wind/solar generation forecast for the DE-LU bidding zone, from 2019-01-01 onward. Pulled via the ENTSO-E API using a token loaded from `.env`.

All series are merged into a single hourly panel: `de_lu_features.parquet`, 19 columns, timezone aware (`Europe/Brussels`), zero nulls, starting 2019-01-01. This is the panel the production model is trained on.

**Outage events, EEX Group Transparency API (REMIT/UMM).** A full 5 year backfill of unplanned power plant outage messages (987,439 events) scraped from `api.eex-group.com`. This dataset was engineered into hourly features and tested as a candidate addition to the model.

## Model

**HistGradientBoostingRegressor** (scikit-learn), trained as a **quantile regression ensemble** producing five quantiles (q05, q25, q50, q75, q95) per hour, wrapped in **conformalized quantile regression (CQR)** for calibrated prediction intervals. The target is the price's deviation from its own trailing 168 hour mean, this keeps the model from having to extrapolate outside the price range it was trained on, which matters most exactly during a spike.

Validation is a **walk forward backtest over 49 monthly blocks**: the model is retrained at each step using only data available up to that point in time, so the reported accuracy reflects what would actually have been seen in production, not a model trained with hindsight.

### Results (production model, full backtest)

| | n hours | median MAE | RMSE | pinball (avg) |
|---|---|---|---|---|
| Overall | 34,365 | 30.1 | 48.2 | 10.25 |
| Calm hours (< 361 EUR/MWh) | 32,646 | 27.4 | 43.5 | 9.30 |
| Spike hours (≥ 361 EUR/MWh) | 1,719 | 81.4 | 102.9 | 28.25 |

**Calibration** (empirical vs. nominal coverage per quantile):

| Quantile | Nominal | Empirical |
|---|---|---|
| q05 | 0.05 | 0.070 |
| q25 | 0.25 | 0.276 |
| q50 | 0.50 | 0.482 |
| q75 | 0.75 | 0.692 |
| q95 | 0.95 | 0.917 |

90% interval (q05–q95): nominal 0.90, empirical 0.847.

The lower and middle quantiles are well calibrated. q75/q95 still run a little hot, the model's upper tail band is slightly narrower than it should be. This gap is the reason Feature testing (below) and the current forward looking work exist.

**What drives the model** (permutation importance, median model, Δ MAE in EUR/MWh when a feature is shuffled):

| Feature | Importance |
|---|---|
| `residual_load` (load minus renewable forecast) | 24.40 |
| `price_roll168_mean` | 4.74 |
| `price_roll168_std` | 1.75 |
| `hour` | 0.61 |
| `spike_freq_168` (rolling 168h spike frequency, shifted 24h) | 0.50 |
| `price_lag_24` | 0.50 |
| everything else (calendar, wind/solar forecast) | ≤ 0.19 |

`residual_load` dominates by an order of magnitude, Wind/solar forecast components individually score near zero, which reflects collinearity with `residual_load` rather than irrelevance.

An early version of this model had a real calibration bug (q50 empirical coverage 0.39 vs. nominal 0.50), traced to symmetric CQR applying one shared correction across quantiles that were actually miscalibrated by different amounts in different directions. Replacing it with a **per quantile shift**, computed independently for each of the five quantiles, is what's reflected in the calibration table above.

## Testing outage data (REMIT/UMM)
 
The hypothesis: unplanned generator outages should carry information about upcoming price spikes, since losing dispatchable capacity tightens the market exactly when prices are most likely to move. Outage events were engineered into an hourly feature, total unplanned dispatchable capacity offline, and added to a second backtest run under the same walk-forward methodology.
 
| | Baseline | + outage feature |
|---|---|---|
| Overall median MAE | 30.1 | 29.9 |
| Overall pinball (avg) | 10.25 | 10.17 |
| Spike-hour median MAE | 81.4 | **80.0** |
| Spike-hour RMSE | 102.9 | **101.6** |
| Spike-hour pinball (avg) | 28.25 | **27.62** |
| q75 calibration | 0.692 | 0.692 |
| q95 calibration | 0.917 | 0.920 |
 
The improvement shows up specifically where it matters: spike hour accuracy and pinball loss both improved. Permutation importance backs this up, the feature scores 0.18.
 
**Conclusion: outage data is part of the production model.**

## Risk module

This module answers the questions a risk manager or clearing house would ask about the forecasted prices: how much could be lost on a bad day, how bad is a truly extreme day, and how much collateral should be held against that risk. All figures below refer to a position of 1 MWh and are expressed in €/MWh; the dashboard's position slider scales them linearly to any position size in EUR notional terms.

### Value at Risk (VaR) and Expected Shortfall (ES)

VaR estimates the daily loss that should only be exceeded 1 day in 100 (the 99% confidence level). ES goes one step further and asks: *on* those worst days, how bad is the loss on average?

Both are computed two ways:

- **Historical** read directly from the model's own predictive distribution.
- **Parametric** under the textbook assumption that returns follow a normal (Gaussian) distribution.

On a real 2022 test window, the historical VaR came out at **€226/MWh**, while the parametric figure was roughly 15% lower. That gap is the point: power prices have far more extreme days than a normal distribution allows for, so the textbook assumption understates risk exactly where it matters most. Expected Shortfall on the same window was **€285.5/MWh** — meaning that when one of those 1 in 100 days does occur, the average loss is €285.5 per MWh held.

### Initial margin

This is the collateral a clearing house would require upfront to cover potential losses while it closes out a defaulted position, a process that takes several days, not one. The one day VaR is therefore scaled up to cover that multi day window, giving a margin requirement of **€319.6/MWh** on the 2022 test period.

### Procyclicality

The margin figure is recomputed daily using a rolling one year lookback of price history. This reveals a well known problem with how initial margin works in practice: after a price spike, the spike stays inside the lookback window for a full year, keeping collateral requirements elevated long after the event and, worse, margin requirements rise precisely when markets are already stressed and participants can least afford to post more collateral. 

| Metric (99%, 2022 test window) | Value |
|---|---|
| Historical VaR | €226/MWh |
| Expected Shortfall | €285.5/MWh |
| Initial margin | €319.6/MWh |

## The dashboard

The dashboard is a **static, client side page**, all figures are pre computed from the backtest, nothing is calculated live in the browser.
