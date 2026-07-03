"""
Phase 2 risk core: VaR, Expected Shortfall, and initial margin.

Design choices that are deliberate (and defensible in interview):

1. We work in PRICE CHANGES (EUR/MWh P&L), not log returns.
   Day-ahead power prices go negative and through zero, so log returns
   are undefined/unstable. A position's daily P&L is
   position_MWh * (price_t - price_{t-1}); risk is a quantile of that.

2. Every function takes the SAME price frame the pipeline emits, so the
   module is data-source agnostic - synthetic now, real parquet later.

3. Initial margin is a high-quantile loss scaled to a margin period of
   risk (MPOR). This is the simple, transparent (SPAN-lite / filtered-
   historical) IM - the version you can fully explain, which matters
   more to a clearing-house audience than an opaque one.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

PRICE_COL = "price_da"


# --------------------------------------------------------------------------
# P&L construction
# --------------------------------------------------------------------------
def price_changes(
    prices: pd.DataFrame,
    *,
    price_col: str = PRICE_COL,
    horizon_hours: int = 24,
) -> pd.Series:
    """Day-over-day price changes in EUR/MWh.

    horizon_hours=24 gives the daily P&L driver for a 1-day position on
    the hourly series. Returns a clean (NaN-dropped) Series.
    """
    s = prices[price_col].astype(float)
    return s.diff(horizon_hours).dropna()


def position_pnl(
    prices: pd.DataFrame,
    position_mwh: float,
    *,
    price_col: str = PRICE_COL,
    horizon_hours: int = 24,
) -> pd.Series:
    """P&L of a flat position (EUR). Positive position = long power."""
    return position_mwh * price_changes(
        prices, price_col=price_col, horizon_hours=horizon_hours
    )


# --------------------------------------------------------------------------
# VaR / ES  (loss-positive convention: VaR > 0 means a loss)
# --------------------------------------------------------------------------
def historical_var(pnl: pd.Series, alpha: float = 0.99) -> float:
    """Historical VaR at confidence alpha. Returned as a positive loss."""
    return float(-np.quantile(pnl, 1.0 - alpha))


def historical_es(pnl: pd.Series, alpha: float = 0.99) -> float:
    """Historical Expected Shortfall (CVaR): mean loss beyond VaR."""
    cutoff = np.quantile(pnl, 1.0 - alpha)
    tail = pnl[pnl <= cutoff]
    return float(-tail.mean()) if len(tail) else float("nan")


def parametric_var(pnl: pd.Series, alpha: float = 0.99) -> float:
    """Gaussian (variance-covariance) VaR. Useful as a sanity baseline;
    will UNDERstate tails vs historical on spiky power data - which is
    itself a point worth showing."""
    from scipy.stats import norm  # local import keeps the core dep-light
    mu, sigma = pnl.mean(), pnl.std(ddof=1)
    return float(-(mu + sigma * norm.ppf(1.0 - alpha)))


# --------------------------------------------------------------------------
# Initial margin
# --------------------------------------------------------------------------
def initial_margin(
    pnl: pd.Series,
    *,
    alpha: float = 0.99,
    mpor_days: int = 2,
) -> float:
    """Quantile-based initial margin, scaled to the margin period of risk.

    IM = historical VaR(alpha) on daily P&L, scaled by sqrt(MPOR).
    sqrt-time scaling assumes ~iid daily P&L over the close-out window;
    it is the standard transparent first cut. (A filtered-historical or
    full-revaluation IM would refine this; the interface stays the same.)
    """
    daily_var = historical_var(pnl, alpha=alpha)
    return daily_var * np.sqrt(mpor_days)


def rolling_initial_margin(
    pnl: pd.Series,
    *,
    window: int = 250 * 24,   # ~1y of hourly obs
    alpha: float = 0.99,
    mpor_days: int = 2,
    step: int = 24,           # recompute daily, not every hour
) -> pd.Series:
    """Initial margin recomputed on a rolling lookback window.

    This is the procyclicality engine: as the lookback fills with the
    stressed regime, IM ramps up - margin tightening into the crisis,
    exactly when members are least able to fund it.
    """
    out_idx, out_val = [], []
    for end in range(window, len(pnl) + 1, step):
        win = pnl.iloc[end - window : end]
        out_idx.append(pnl.index[end - 1])
        out_val.append(initial_margin(win, alpha=alpha, mpor_days=mpor_days))
    return pd.Series(out_val, index=out_idx, name="initial_margin")


# --------------------------------------------------------------------------
# Convenience: one-shot risk summary for a position over a price frame
# --------------------------------------------------------------------------
def risk_summary(
    prices: pd.DataFrame,
    position_mwh: float = 1.0,
    *,
    price_col: str = PRICE_COL,
    alpha: float = 0.99,
    mpor_days: int = 2,
) -> dict:
    pnl = position_pnl(prices, position_mwh, price_col=price_col)
    return {
        "alpha": alpha,
        "position_mwh": position_mwh,
        "hist_var": historical_var(pnl, alpha),
        "hist_es": historical_es(pnl, alpha),
        "initial_margin": initial_margin(pnl, alpha=alpha, mpor_days=mpor_days),
        "n_obs": len(pnl),
    }
