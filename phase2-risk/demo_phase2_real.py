"""
Follow-up diagnostic: directly compare the pre-dip and post-dip 120-day
windows to see what observation(s) rolled OUT, vs. what rolled IN.

Run: python inspect_july_dip_v2.py
"""

import pandas as pd
import risk

PRICE_COL = "price"

df = pd.read_parquet("de_lu_features.parquet", columns=[PRICE_COL])
df = df.dropna(subset=[PRICE_COL])

year_2022 = df.loc["2022-01-01":"2022-12-31"]
pnl = risk.position_pnl(year_2022, 1.0, price_col=PRICE_COL)

window_hours = 120 * 24

# Window ending 2022-07-09 (IM = 264.9) vs window ending 2022-07-10 (IM = 253.2)
end_before = pd.Timestamp("2022-07-09 00:00:00+02:00")
end_after = pd.Timestamp("2022-07-10 00:00:00+02:00")

# Locate positions in pnl to slice by count, matching rolling_initial_margin's
# iloc-based windowing exactly.
pos_before = pnl.index.get_indexer([end_before], method="nearest")[0]
pos_after = pnl.index.get_indexer([end_after], method="nearest")[0]

win_before = pnl.iloc[pos_before - window_hours + 1 : pos_before + 1]
win_after = pnl.iloc[pos_after - window_hours + 1 : pos_after + 1]

print(f"Window BEFORE: {win_before.index.min()} to {win_before.index.max()}  (n={len(win_before)})")
print(f"Window AFTER:  {win_after.index.min()} to {win_after.index.max()}  (n={len(win_after)})")

# Observations present in the "before" window but NOT in "after" (rolled out)
rolled_out = win_before.index.difference(win_after.index)
rolled_in = win_after.index.difference(win_before.index)

print(f"\n{len(rolled_out)} obs rolled OUT, worst ones:")
print(pnl.loc[rolled_out].sort_values().head(10))

print(f"\n{len(rolled_in)} obs rolled IN, worst ones:")
print(pnl.loc[rolled_in].sort_values().head(10))

# Direct VaR comparison to confirm the mechanism
print(f"\nVaR(before window) = {risk.historical_var(win_before, 0.99):.1f}")
print(f"VaR(after window)  = {risk.historical_var(win_after, 0.99):.1f}")