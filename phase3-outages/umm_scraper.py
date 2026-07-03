"""
Phase 3 - REMIT UMM historical backfill scraper.

Pulls Germany power non-availability events (planned + unplanned outages)
from the EEX Transparency Platform's underlying JSON API - the same data
the public UMM table at eex-transparency.com/power/de/umms renders, just
fetched directly instead of scraping rendered HTML (the page is JS-rendered,
so a plain HTML scraper would get nothing).

Endpoint discovered via browser DevTools Network tab:
    https://api.eex-group.com/pub/transparency/non-availability-events/Power
    ?reportedDateFrom=YYYY-MM-DD&reportedDateTo=YYYY-MM-DD&country=DE

Design choices (mirrors data_pipeline.py):
  - Weekly chunks (matches the site's own default 7-day view), cached to
    disk so a crash/interrupt doesn't lose progress and dev iteration
    doesn't re-hit the endpoint.
  - This is a scraped endpoint, not a paid/documented API - so we're more
    conservative than the ENTSO-E client: explicit delay between requests,
    a browser-like User-Agent, and we stop cleanly rather than hammering
    on repeated failures.
  - Response shape is {"header": [...], "data": [[...], [...]]} - a table,
    not records - so we zip header+row into dicts ourselves.

Run: python umm_scraper.py
"""

from __future__ import annotations
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.eex-group.com/pub/transparency/non-availability-events/Power"
COUNTRY = "DE"
CACHE_DIR = Path("umm_cache")
REQUEST_DELAY_SECONDS = 1.5  # be polite - this isn't a documented public API
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.eex-transparency.com/",
    "Origin": "https://www.eex-transparency.com",
}

def week_chunks(start: date, end: date):
    """Yield (chunk_start, chunk_end) weekly date pairs covering [start, end]."""
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=6), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def fetch_week(start: date, end: date) -> pd.DataFrame | None:
    """Fetch one weekly chunk, with on-disk caching and retry-with-backoff."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"umm_{start.isoformat()}_{end.isoformat()}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    params = {
        "reportedDateFrom": start.isoformat(),
        "reportedDateTo": end.isoformat(),
        "country": COUNTRY,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            header = payload.get("header", [])
            rows = payload.get("data", [])

            if not rows:
                print(f"  [empty] {start} to {end}: no events")
                df = pd.DataFrame(columns=header)
            else:
                df = pd.DataFrame(rows, columns=header)
                # API returns loosely-typed JSON - columns like `remarks`
                # are mostly null with occasional free-text (e.g. German
                # outage-reason strings), so pandas mis-infers them as
                # int64 and pyarrow chokes on the first real string.
                # Force object-dtype columns to nullable string dtype
                # before writing so parquet doesn't have to guess.
                for col in df.select_dtypes(include="object").columns:
                    df[col] = df[col].astype("string")

            df.to_parquet(cache_file)
            time.sleep(REQUEST_DELAY_SECONDS)
            return df

        except requests.exceptions.RequestException as exc:
            wait = 5 * (attempt + 1)
            print(f"  [retry {attempt + 1}/{MAX_RETRIES}] {start} to {end}: {exc} -> wait {wait}s")
            time.sleep(wait)

    print(f"  [fail] gave up on {start} to {end}")
    return None


def backfill(start: date, end: date) -> pd.DataFrame:
    """Fetch all weekly chunks in [start, end] and concatenate into one frame."""
    parts = []
    chunks = list(week_chunks(start, end))
    print(f"Fetching {len(chunks)} weekly chunks from {start} to {end}...")

    for i, (cs, ce) in enumerate(chunks, 1):
        print(f"[{i}/{len(chunks)}] {cs} to {ce}")
        df = fetch_week(cs, ce)
        if df is not None and len(df):
            parts.append(df)

    if not parts:
        raise SystemExit("No data fetched for any week - check the endpoint/params.")

    full = pd.concat(parts, ignore_index=True)

    # Dedup: the API includes each event's FULL history on every request that
    # overlaps its reported window (same pattern as ENTSO-E year-seam dupes),
    # so a single unplanned outage reported once can appear in multiple
    # weekly chunks. messageID + modified should uniquely identify a
    # specific version of a specific message.
    before = len(full)
    if "messageID" in full.columns and "modified" in full.columns:
        full = full.drop_duplicates(subset=["messageID", "modified"])
    print(f"\nTotal rows: {before} -> {len(full)} after dedup")

    return full


if __name__ == "__main__":
    START = date(2019, 1, 1)  # match de_lu panel start — see data_pipeline.py
    END = date.today()

    events = backfill(START, END)
    events.to_parquet("umm_events_raw.parquet")
    print(f"\nsaved -> umm_events_raw.parquet {events.shape}")