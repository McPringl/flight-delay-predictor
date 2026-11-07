"""
Step 1: Download US BTS (Bureau of Transportation Statistics) on-time data.

The BTS publishes one zipped CSV per month ("PREZIP" files). This script:
  1. downloads each month for the years we choose,
  2. keeps only the columns the model needs (saves memory and disk),
  3. stacks everything into ONE parquet file: data/flights_raw.parquet

Run:  python download_data.py
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

# --- Settings -----------------------------------------------------------------
YEARS = [2023, 2024]           # two recent years (your choice)
DATA_DIR = Path(__file__).parent / "data"
OUT_FILE = DATA_DIR / "flights_raw.parquet"

# Public BTS URL pattern. {year} and {month} are filled in for each file.
URL_TEMPLATE = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

# Columns we keep from the ~110 BTS columns.
# NOTE: ArrDel15 (arrived 15+ min late) is our prediction TARGET.
# We deliberately do NOT keep actual departure/arrival delay columns as features,
# because those are only known AFTER the flight (that would be "data leakage").
KEEP_COLUMNS = [
    "Year", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
    "Reporting_Airline", "Flight_Number_Reporting_Airline",
    "Origin", "Dest", "CRSDepTime", "CRSArrTime",
    "Distance", "CRSElapsedTime",
    "Cancelled", "Diverted",
    "ArrDel15",   # TARGET: 1 = arrived 15+ minutes late, 0 = on time
]


def download_month(year: int, month: int) -> pd.DataFrame:
    """Download one month from BTS and return it as a DataFrame with KEEP_COLUMNS."""
    url = URL_TEMPLATE.format(year=year, month=month)
    print(f"  downloading {year}-{month:02d} ...", end=" ", flush=True)

    response = requests.get(url, timeout=300)
    response.raise_for_status()

    # The zip holds exactly one CSV. Open it straight from memory (no temp files).
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_name = [name for name in zf.namelist() if name.lower().endswith(".csv")][0]
        with zf.open(csv_name) as f:
            # usecols as a function: silently skip any column BTS did not include
            df = pd.read_csv(f, usecols=lambda c: c in KEEP_COLUMNS, low_memory=False)

    print(f"{len(df):,} rows")
    return df


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    frames = []

    for year in YEARS:
        for month in range(1, 13):
            frames.append(download_month(year, month))

    flights = pd.concat(frames, ignore_index=True)
    flights.to_parquet(OUT_FILE, index=False)

    print(f"\nSaved {len(flights):,} flights to {OUT_FILE}")
    print("Columns:", list(flights.columns))


if __name__ == "__main__":
    main()
