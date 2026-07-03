"""
Synthetic DE_LU day-ahead price generator.

Emits the SAME dataframe contract as data_pipeline.py:
  - tz-aware DatetimeIndex (Europe/Brussels), hourly
  - one price column (default 'price_da'), in EUR/MWh

So every function downstream (VaR/ES/IM) can be developed and tested
offline now, then run unchanged once the real parquet cache lands.

Two regimes:
  - 'calm'     : mean-reverting around a stable level, modest vol
  - 'stressed' : level ramp + rising vol (a stylised gas/Hormuz-type
                 shock), used to demonstrate procyclical margin
"""

from __future__ import annotations
import numpy as np
import pandas as pd

PRICE_COL = "price_da"
TZ = "Europe/Brussels"


def generate_synthetic_prices(
    start: str = "2022-01-01",
    end: str = "2022-12-31",
    regime: str = "calm",
    *,
    price_col: str = PRICE_COL,
    base_level: float = 60.0,      # EUR/MWh, calm anchor
    seed: int | None = 42,
) -> pd.DataFrame:
    """Return an hourly day-ahead price frame matching the pipeline contract.

    The series is an Ornstein-Uhlenbeck mean-reverting process with a
    daily/seasonal shape, fat-tailed spike innovations, and (in the
    stressed regime) a time-varying level and volatility.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, end=end, freq="h", tz=TZ)
    n = len(idx)

    # Deterministic shape: diurnal + weekly, scaled to EUR/MWh.
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    diurnal = 18.0 * np.sin((hour - 6) / 24 * 2 * np.pi)      # morning/evening peaks
    weekly = np.where(dow >= 5, -8.0, 0.0)                    # weekend softness

    # Regime-dependent level and volatility paths.
    t = np.linspace(0.0, 1.0, n)
    if regime == "calm":
        level = np.full(n, base_level)
        vol = np.full(n, 8.0)
    elif regime == "stressed":
        # Level ramps ~4x, vol ramps ~5x over the window: a fuel-shock path.
        level = base_level * (1.0 + 3.0 * t**1.5)
        vol = 8.0 * (1.0 + 4.0 * t**2)
    else:
        raise ValueError(f"unknown regime {regime!r}; use 'calm' or 'stressed'")

    # Mean-reverting (OU) core with fat-tailed spike innovations.
    theta = 0.15                     # reversion speed
    p = np.empty(n)
    p[0] = level[0]
    for i in range(1, n):
        anchor = level[i] + diurnal[i] + weekly[i]
        # Student-t innovations -> fatter tails than Gaussian (spikes).
        shock = vol[i] * rng.standard_t(df=4) / np.sqrt(4 / 2)
        p[i] = p[i - 1] + theta * (anchor - p[i - 1]) + shock

    # Occasional positive price spikes (scarcity hours).
    spike_mask = rng.random(n) < 0.004
    p = p + spike_mask * rng.gamma(shape=2.0, scale=80.0, size=n)

    # Power prices CAN go negative (renewables surplus) - do not clip at 0.
    p = np.clip(p, -100.0, 4000.0)

    return pd.DataFrame({price_col: p}, index=idx)
