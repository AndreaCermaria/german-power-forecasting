# check_umm_filters.py
import pandas as pd

df = pd.read_parquet("umm_events_raw.parquet")

print(f"Total raw events: {len(df)}\n")

print("=== unavailabilityType values ===")
print(df["unavailabilityType"].value_counts(dropna=False))

print("\n=== eventType values ===")
print(df["eventType"].value_counts(dropna=False))

print("\n=== biddingZone values (top 15) ===")
print(df["biddingZone"].value_counts(dropna=False).head(15))

# Specifically: is DE_LU represented under a different code/label?
DE_ZONE = "10Y1001A1001A82H"
print(f"\n=== rows matching DE_ZONE exactly ({DE_ZONE}) ===")
print((df["biddingZone"] == DE_ZONE).sum())

print("\n=== any biddingZone containing 'DE' (string match, case-insensitive) ===")
de_like = df["biddingZone"].astype(str).str.contains("DE", case=False, na=False)
print(df.loc[de_like, "biddingZone"].value_counts())

# Final: how many rows survive the exact filter combination used in load_and_filter()
mask = (
    (df["unavailabilityType"] == "Unplanned")
    & (df["eventType"] == "Production unavailability")
    & (df["biddingZone"] == DE_ZONE)
)
print(f"\n=== rows passing ALL THREE filters ===")
print(mask.sum())