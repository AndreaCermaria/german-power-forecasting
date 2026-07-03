"""
One-off cleanup: drop the unplanned_mw_offline_total column from the panel.

Decision: unplanned_mw_offline_dispatchable carries real signal (permutation
importance ~0.09, above the renewable-collinearity noise floor), while
unplanned_mw_offline_total does not (importance ~-0.04, below the noise
floor) — it conflates renewable and dispatchable outages, which dilutes any
signal from the dispatchable component that actually matters for price.

Run: python drop_total_col.py
"""

import pandas as pd

PANEL_PATH = "de_lu_features.parquet"
DROP_COL = "unplanned_mw_offline_total"

panel = pd.read_parquet(PANEL_PATH)

if DROP_COL not in panel.columns:
    print(f"'{DROP_COL}' not in panel — already dropped, nothing to do.")
else:
    panel = panel.drop(columns=[DROP_COL])
    panel.to_parquet(PANEL_PATH)
    print(f"dropped '{DROP_COL}' -> panel shape: {panel.shape}")