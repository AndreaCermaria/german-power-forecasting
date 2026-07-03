"""
merge_umm_features.py

Left-joins UMM outage features onto the main panel. Writes a new file
rather than overwriting de_lu_features.parquet, so the original stays
reproducible/diffable and forecast.py can be pointed at either via
--parquet.
"""

import pandas as pd

panel = pd.read_parquet("de_lu_features.parquet")
umm = pd.read_parquet("umm_features.parquet")

# drop the previous (uncorrected) UMM columns before rejoining
stale = ["unplanned_mw_offline_total", "unplanned_mw_offline_dispatchable"]
panel = panel.drop(columns=[c for c in stale if c in panel.columns])

before_cols = set(panel.columns)
panel = panel.join(umm, how="left")
new_cols = set(panel.columns) - before_cols
print(f"Added columns: {new_cols}")
print(f"Panel shape: {panel.shape}")
print(f"Nulls introduced: {panel[list(new_cols)].isna().sum().to_dict()}")

panel.to_parquet("de_lu_features.parquet")