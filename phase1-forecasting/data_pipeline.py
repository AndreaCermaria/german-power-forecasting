"""
Phase 1 data pipeline — DE_LU day-ahead power price forecasting.

Responsibilities (all upstream of any modelling):
  1. Fetch the three core ENTSO-E series, one year at a time, CACHED to disk.
     (wind+solar forecast expands to Solar / Wind Onshore / Wind Offshore.)
  2. Merge into one clean hourly panel (DST-safe).
  3. Engineer residual load, calendar, and leakage-aware lag features.
  4. VALIDATE the panel so you trust it before modelling.

Setup (never commit the token):
    # put this in a .env file at the repo root, and add .env to .gitignore
    echo 'ENTSOE_API_KEY=<your-token>' > .env
    echo '.env' >> .gitignore

    pip install entsoe-py pandas pyarrow holidays python-dotenv
    python data_pipeline.py

Design notes:
  - entsoe-py methods are year-limited internally, but we still loop per year
    so each year is cached and dev iterations don't re-hit the API.
  - Per-year windows are half-open in intent but the API returns the boundary
    hour on both sides, so Jan-01 00:00 is duplicated at every year seam. We
    drop those duplicates per-series BEFORE the axis=1 join (concat(axis=1)
    requires a unique index on every input or it raises InvalidIndexError).
  - ENTSO-E returns HTTP 200 even when a window has no data; the library turns
    that into NoMatchingDataError, which we catch so one empty year can't kill
    the whole pull.
  - Everything stays tz-aware in Europe/Brussels so the 23h/25h DST days are
    handled correctly instead of silently corrupting the hourly grid.
"""

from __future__ import annotations
import os
import time
from pathlib import Path

import pandas as pd

ZONE = "DE_LU"
TZ = "Europe/Brussels"
CACHE_DIR = Path("data_cache")

# logical series name -> entsoe-py client method
SERIES = {
    "price":   "query_day_ahead_prices",
    "load_fc": "query_load_forecast",
    "ren_fc":  "query_wind_and_solar_forecast",
}


# --------------------------------------------------------------------------- #
# Credentials                                                                 #
# --------------------------------------------------------------------------- #
def load_api_key() -> str:
    """Read ENTSOE_API_KEY from the environment, hydrating from .env if present."""
    try:
        from dotenv import load_dotenv
        load_dotenv()  # no-op if there is no .env file
    except ModuleNotFoundError:
        pass  # python-dotenv optional; plain env var still works

    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise SystemExit(
            "ENTSOE_API_KEY is not set. Put it in a .env file (and add .env to "
            ".gitignore) or `export ENTSOE_API_KEY=...`. Never commit the token."
        )
    return key


# --------------------------------------------------------------------------- #
# Fetch layer (API-dependent)                                                 #
# --------------------------------------------------------------------------- #
def _canonical(obj, name: str) -> pd.DataFrame:
    """Normalise a series to its logical column name.

    entsoe-py is inconsistent: query_day_ahead_prices returns a Series, while
    query_load_forecast returns a 1-col DataFrame ("Forecasted Load"). Both
    should become a single column named `name`. Multi-column frames (the
    wind+solar forecast: Solar / Wind Onshore / Wind Offshore) keep their
    descriptive names so add_features can detect them by substring.
    """
    if isinstance(obj, pd.Series):
        return obj.rename(name).to_frame()
    if obj.shape[1] == 1:
        return obj.set_axis([name], axis=1)
    return obj


def fetch_year(client, name: str, year: int) -> pd.DataFrame | None:
    """Fetch one series for one calendar year, with on-disk caching."""
    from entsoe.exceptions import NoMatchingDataError

    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"{name}_{year}.parquet"
    if cache.exists():
        # Normalise on read too: caches written before the column fix still hold
        # the raw API names, so this repairs them without re-hitting the API.
        return _canonical(pd.read_parquet(cache), name)

    method = getattr(client, SERIES[name])
    start = pd.Timestamp(f"{year}0101", tz=TZ)
    end = pd.Timestamp(f"{year + 1}0101", tz=TZ)

    for attempt in range(3):  # ENTSO-E has intermittent 5xx; retry with backoff
        try:
            raw = method(ZONE, start=start, end=end)
            df = _canonical(raw, name)
            df.to_parquet(cache)
            return df
        except NoMatchingDataError:
            print(f"  [skip] no data for {name} {year}")
            return None
        except Exception as exc:  # noqa: BLE001 - want to retry on transient errors
            wait = 5 * (attempt + 1)
            print(f"  [retry {attempt + 1}/3] {name} {year}: {exc} -> wait {wait}s")
            time.sleep(wait)
    print(f"  [fail] gave up on {name} {year}")
    return None


def fetch_panel(years: range, api_key: str) -> pd.DataFrame:
    """Fetch all series across all years and merge into one hourly panel."""
    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=api_key)
    per_series: dict[str, pd.DataFrame] = {}
    for name in SERIES:
        parts = [fetch_year(client, name, y) for y in years]
        parts = [p for p in parts if p is not None]
        if parts:
            s = pd.concat(parts).sort_index()
            # Year windows overlap at Jan-01 00:00, so each seam is duplicated.
            # Drop before the axis=1 join. Safe: index is tz-aware Europe/Brussels,
            # so the October DST fall-back hour is two distinct UTC instants and is
            # NOT flagged here — we only collapse genuine boundary repeats.
            s = s[~s.index.duplicated(keep="first")]
            per_series[name] = s

    if not per_series:
        raise SystemExit("No data fetched for any series — check API key / network.")

    panel = pd.concat(per_series.values(), axis=1, sort=False)
    panel = panel.tz_convert(TZ).sort_index()
    # collapse to a clean hourly grid (load/renewables may arrive at 15-min MTU)
    panel = panel.resample("1h").mean()
    return panel


# --------------------------------------------------------------------------- #
# Transform layer (pure pandas - unit tested offline)                         #
# --------------------------------------------------------------------------- #
def add_features(panel: pd.DataFrame, country_holidays=None) -> pd.DataFrame:
    """Engineer residual load, calendar, and leakage-aware lags."""
    df = panel.copy()

    # Residual load = forecast demand minus forecast renewables. THE driver.
    ren_cols = [c for c in df.columns if ("Solar" in c) or ("Wind" in c)]
    df["renewables_fc"] = df[ren_cols].sum(axis=1)
    df["residual_load"] = df["load_fc"] - df["renewables_fc"]

    # Calendar
    idx = df.index
    df["hour"] = idx.hour
    df["dow"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    if country_holidays is None:
        import holidays
        country_holidays = holidays.Germany()
    df["is_holiday"] = pd.Series(idx.date, index=idx).isin(country_holidays).astype(int)

    # Lags: only price actuals from D-1 or earlier are known at gate closure.
    # 24h = same hour yesterday, 168h = same hour last week.
    df["price_lag_24"] = df["price"].shift(24)
    df["price_lag_168"] = df["price"].shift(168)
    df["price_roll168_mean"] = df["price"].shift(24).rolling(168).mean()

    # Volatility features: recent dispersion of price, lag-safe (>=24h shift so
    # nothing past gate closure leaks in). These give the QUANTILE models a
    # "how turbulent is the market right now" signal, letting them widen their
    # intervals in choppy regimes — the targeted fix for too-narrow prediction
    # intervals (calibration). The weekly window matches the level reference; the
    # short window reacts faster when volatility spikes.
    df["price_roll168_std"] = df["price"].shift(24).rolling(168).std()
    df["price_roll24_std"] = df["price"].shift(24).rolling(24).std()
    price_shifted = df["price"].shift(24)
    spike_flag = (price_shifted > (df["price_roll168_mean"] + 2 * df["price_roll168_std"])).astype(int)
    df["spike_freq_168"] = spike_flag.rolling(168).mean()
    return df


# --------------------------------------------------------------------------- #
# Validation layer                                                            #
# --------------------------------------------------------------------------- #
def validate(panel: pd.DataFrame) -> dict:
    """Cheap sanity checks. Returns a report dict; prints a readable summary."""
    report = {}
    report["rows"] = len(panel)
    report["span"] = (panel.index.min(), panel.index.max())
    report["dup_timestamps"] = int(panel.index.duplicated().sum())

    # The resample() rebuilds a contiguous grid, so this is a structural check:
    # it should be 0. Genuine data gaps surface as empty_hours / nan_pct below.
    if len(panel) > 1:
        expected = len(pd.date_range(panel.index.min(), panel.index.max(),
                                     freq="1h", tz=TZ))
        report["grid_shortfall"] = expected - report["rows"]

    # Real gaps: hours where every raw input series is NaN.
    core = [c for c in ("price", "load_fc") if c in panel]
    if core:
        report["empty_hours"] = int(panel[core].isna().all(axis=1).sum())

    report["nan_pct"] = (panel.isna().mean() * 100).round(2).to_dict()

    if "price" in panel:
        p = panel["price"].dropna()
        report["price_min"] = float(p.min())
        report["price_max"] = float(p.max())
        report["negative_price_hours"] = int((p < 0).sum())  # real in DE, keep them
    if "residual_load" in panel:
        report["residual_load_negative_hours"] = int((panel["residual_load"] < 0).sum())

    print("=== panel validation ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    return report


if __name__ == "__main__":
    years = range(2019, 2025)
    panel = fetch_panel(years, load_api_key())
    feats = add_features(panel)
    validate(feats)
    feats.to_parquet("de_lu_features.parquet")
    print("\nsaved -> de_lu_features.parquet", feats.shape)