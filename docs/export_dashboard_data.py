"""
Export backtest_predictions.parquet into a compact JSON file for the static
HTML dashboard (docs/index.html). Run this locally whenever the backtest is
re-run, then commit docs/data/backtest.json alongside the updated dashboard.

Run: python export_dashboard_data.py
"""

import json
import os

import pandas as pd

SRC = "backtest_predictions.parquet"
OUT = "docs/data/backtest.json"
COLS = ["q05", "q25", "q50", "q75", "q95", "actual"]


def main():
    df = pd.read_parquet(SRC).sort_index()
    df = df[COLS].round(2)

    ts_ms = (df.index.asi8 // 10**6).tolist()
    rows = df.values.tolist()
    data = [[t] + r for t, r in zip(ts_ms, rows)]

    payload = {"columns": ["ts"] + COLS, "data": data}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_kb = os.path.getsize(OUT) / 1024
    print(f"wrote {OUT}  ({len(data):,} rows, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()