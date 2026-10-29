"""
Expand Citi Bike Jan-2021 (1,000,000-row sample) into a simulated full calendar year 2021  (v2).

Why v2: v1 cut the time-ordered file into 365 slices, so each simulated day only contained ~2 hours of
original time-of-day (87% of hours had zero trips). v2 keeps whole-day structure instead.

Method (weekday-aligned day mapping):
  1. Drop rows with started_at before 2021-01-01 (out-of-range outliers).
  2. Original January day d (1..31) belongs to cycle position p = (d-1) mod 28.
     1 Jan 2021 and 29 Jan 2021 are both Fridays, so 28 days preserves day-of-week.
  3. Simulated day n (0..364 = 1 Jan..31 Dec 2021) takes its trips from position p = n mod 28.
     All trips of a position are randomly (seed 42) and evenly shared among the simulated days of that position,
     so every original trip is used exactly once (no duplicated trips / ride_id).
  4. A trip keeps its original time-of-day and duration; only the calendar date changes.
     Original timestamps are kept in orig_started_at / orig_ended_at.
LIMITATION: dates are simulated. There is no real seasonality, holiday or weather effect in this series.
"""
import numpy as np
import pandas as pd

SRC = "2021_01_citibike_tripdata.xlsx"                       # original January file
OUT = "citibike_2021_full_year_simulated.csv"

df = pd.read_excel(SRC)
df = df[df["started_at"] >= "2021-01-01"].copy()
df["orig_started_at"], df["orig_ended_at"] = df["started_at"], df["ended_at"]
duration = df["ended_at"] - df["started_at"]
tod = df["started_at"] - df["started_at"].dt.normalize()

pos = ((df["started_at"].dt.day - 1) % 28).to_numpy()
rng = np.random.RandomState(42)
new_day = np.zeros(len(df), dtype=int)
for p in range(28):
    idx = np.where(pos == p)[0]
    days = np.arange(p, 365, 28)                              # simulated days that map to this position
    new_day[idx] = days[rng.permutation(len(idx)) % len(days)]

df["started_at"] = pd.Timestamp("2021-01-01") + pd.to_timedelta(new_day, unit="D") + tod
df["ended_at"] = df["started_at"] + duration
df["is_simulated_date"] = True
df = df.sort_values("started_at").reset_index(drop=True)

# write with full-precision timestamps; DO NOT open/re-save in Excel (it truncates seconds and corrupts ride_id)
df.to_csv(OUT, index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
print(df.shape, df["started_at"].min(), df["started_at"].max())
