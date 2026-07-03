"""
umm_features.py

Builds point-in-time-correct hourly features from REMIT UMM outage data,
aligned to the existing DE_LU panel (de_lu_features.parquet).

Core idea: messageID revisions of the same underlying event are sequenced
by `modified` timestamp. Each revision's stated capacity is only "known"
from its own `modified` time until the next revision supersedes it (or the
event ends) - this prevents leaking later corrections into earlier hours.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

UMM_PATH = "umm_events_raw.parquet"
PANEL_PATH = "de_lu_features.parquet"
DE_ZONE = "10Y1001A1001A82H"
RENEWABLE_FUELS = {"Wind Offshore", "Wind Onshore", "Solar"}


def load_and_filter() -> pd.DataFrame:
    df = pd.read_parquet(UMM_PATH)

    df = df[
        (df["unavailabilityType"] == "Unplanned")
        & (df["eventType"] == "Production unavailability")
        # biddingZone filter removed — the scraper's API query already
        # scopes country=DE; ~25% of valid events had null or TSO-level
        # sub-zone codes and were being silently dropped by an exact
        # top-level EIC match against DE_ZONE.
    ].copy()

    for col in ["eventStart", "eventStop", "modified"]:
        df[col] = pd.to_datetime(df[col], utc=True)

    df["base_event_id"] = df["messageID"].str.rsplit("_", n=1).str[0]

    return df


def build_revision_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """For each base event, sequence revisions and compute the window during
    which each revision's capacity figure was the current known state."""
    df = df.sort_values(["base_event_id", "modified"]).copy()

    df["next_modified"] = df.groupby("base_event_id")["modified"].shift(-1)
    df["next_modified"] = df["next_modified"].fillna(pd.Timestamp.max.tz_localize("UTC"))

    df["effective_start"] = df[["modified", "eventStart"]].max(axis=1)
    df["effective_end"] = df[["eventStop", "next_modified"]].min(axis=1)

    # drop invalid/instantaneous intervals (e.g. a closing revision
    # published after the event already ended)
    df = df[df["effective_start"] < df["effective_end"]]

    return df


def sweep_to_hourly(intervals: pd.DataFrame, panel_index: pd.DatetimeIndex,
                     capacity_col: str = "nonAvailableCapacity") -> pd.Series:
    """Sweep-line: convert intervals to +/- deltas, cumsum, align to panel."""
    starts = pd.DataFrame({
        "ts": intervals["effective_start"],
        "delta": intervals[capacity_col],
    })
    ends = pd.DataFrame({
        "ts": intervals["effective_end"],
        "delta": -intervals[capacity_col],
    })
    deltas = pd.concat([starts, ends], ignore_index=True).sort_values("ts")
    deltas = deltas.groupby("ts", as_index=True)["delta"].sum()

    cum = deltas.cumsum()

    # align sweep-line (irregular timestamps) onto the regular hourly panel
    # index via as-of merge: each panel hour gets the most recent cumulative
    # value at or before that hour
    panel_index_utc = panel_index.tz_convert("UTC")
    aligned = cum.reindex(cum.index.union(panel_index_utc)).sort_index()
    aligned = aligned.ffill().reindex(panel_index_utc)  # no fillna(0.0) here
    aligned.index = panel_index  # restore original tz for the merge back

    return aligned


def build_features() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_PATH)
    events = load_and_filter()
    intervals = build_revision_intervals(events)

    total = sweep_to_hourly(intervals, panel.index)

    dispatchable_intervals = intervals[~intervals["fuelType"].isin(RENEWABLE_FUELS)]
    dispatchable = sweep_to_hourly(dispatchable_intervals, panel.index)

    out = pd.DataFrame({
        "unplanned_mw_offline_total": total,
        "unplanned_mw_offline_dispatchable": dispatchable,
    })

    out = out.clip(lower=0.0)

    # sweep-line cumsum drifts by floating-point noise (~1e-9 to 1e-12 MW)
    # at hours that should be exactly zero outage. Snap anything below a
    # physically meaningless threshold (1 kW) back to a clean 0.0.
    out = out.mask(out < 1e-3, 0.0)

    return out


if __name__ == "__main__":
    features = build_features()
    print(features.describe())
    print()
    print(f"Non-zero hours (total): {(features['unplanned_mw_offline_total'] > 0).mean():.1%}")
    print(f"Non-zero hours (dispatchable): {(features['unplanned_mw_offline_dispatchable'] > 0).mean():.1%}")

    features.to_parquet("umm_features.parquet")
    print("\nsaved -> umm_features.parquet", features.shape)