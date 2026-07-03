from umm_scraper import fetch_week
from datetime import date

df = fetch_week(date(2022, 1, 1), date(2022, 1, 7))
print(df.shape)
print(df.head())