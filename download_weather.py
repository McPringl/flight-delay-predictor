"""
Step 2: Download HOURLY historical weather for the busiest US airports.

Sources (both free, no API key needed):
  * OpenFlights airports.dat  -> latitude / longitude of each airport (IATA code)
  * Open-Meteo archive API    -> hourly weather history for a lat/lon

Times are requested in the AIRPORT'S LOCAL time zone (timezone=auto), because
BTS scheduled departure times (CRSDepTime) are also local clock times.

Output: data/weather_hourly.parquet
  columns: origin (IATA), local_time (naive local datetime, hourly),
           temperature_c, precipitation_mm, wind_kmh, wind_gust_kmh

Run AFTER download_data.py:   python download_weather.py
"""

import io
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
FLIGHTS_FILE = DATA_DIR / "flights_raw.parquet"
OUT_FILE = DATA_DIR / "weather_hourly.parquet"
PARTS_DIR = DATA_DIR / "weather_parts"     # one file per airport, so a crash does not lose work
AIRPORTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

TOP_N_AIRPORTS = 150      # the busiest airports cover most flights; the rest get no weather
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"
HOURLY_VARS = "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m"

# Open-Meteo name -> our short column name
RENAME = {
    "temperature_2m": "temperature_c",
    "precipitation": "precipitation_mm",
    "wind_speed_10m": "wind_kmh",
    "wind_gusts_10m": "wind_gust_kmh",
}


def load_airport_coordinates() -> pd.DataFrame:
    """Return a table: iata, lat, lon for airports that have an IATA code."""
    text = requests.get(AIRPORTS_URL, timeout=60).text
    cols = ["id", "name", "city", "country", "iata", "icao", "lat", "lon",
            "alt", "tz_offset", "dst", "tz", "type", "source"]
    airports = pd.read_csv(io.StringIO(text), header=None, names=cols, na_values=["\\N"])
    airports = airports[airports["iata"].notna()]
    return airports[["iata", "lat", "lon"]].drop_duplicates("iata")


def busiest_origins(n: int) -> list[str]:
    """The n airports with the most departures in our flight data."""
    flights = pd.read_parquet(FLIGHTS_FILE, columns=["Origin"])
    return flights["Origin"].value_counts().head(n).index.tolist()


def fetch_airport_weather(iata: str, lat: float, lon: float) -> pd.DataFrame:
    """Hourly weather history for one airport, in its local time zone."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": HOURLY_VARS,
        "timezone": "auto",          # -> local clock time of the airport
        "wind_speed_unit": "kmh",
    }
    # The free API rate-limits (HTTP 429). Wait longer each time and try again.
    for attempt in range(6):
        response = requests.get(ARCHIVE_URL, params=params, timeout=120)
        if response.status_code != 429:
            break
        wait = 60 * (attempt + 1)
        print(f"    rate limited, waiting {wait}s ...", flush=True)
        time.sleep(wait)
    response.raise_for_status()

    hourly = response.json()["hourly"]
    df = pd.DataFrame(hourly).rename(columns={"time": "local_time", **RENAME})
    df["local_time"] = pd.to_datetime(df["local_time"])   # naive local datetime
    df.insert(0, "origin", iata)
    return df


def main() -> None:
    coords = load_airport_coordinates().set_index("iata")
    origins = busiest_origins(TOP_N_AIRPORTS)

    # Each airport is saved to its own small file as soon as it is downloaded.
    # If the script is stopped (or rate-limited), re-running it skips finished airports.
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    for i, iata in enumerate(origins, start=1):
        part_file = PARTS_DIR / f"{iata}.parquet"
        if part_file.exists():
            continue
        if iata not in coords.index:
            print(f"[{i}/{len(origins)}] {iata}: no coordinates, skipped")
            continue
        row = coords.loc[iata]
        print(f"[{i}/{len(origins)}] {iata} ...", flush=True)
        try:
            fetch_airport_weather(iata, row["lat"], row["lon"]).to_parquet(part_file, index=False)
        except (requests.RequestException, OSError) as err:
            # Network hiccup: skip this airport for now. Re-running the script retries it.
            print(f"    failed ({type(err).__name__}), will retry on next run", flush=True)
            time.sleep(30)
            continue
        time.sleep(3)                 # be polite to the free API

    frames = [pd.read_parquet(p) for p in sorted(PARTS_DIR.glob("*.parquet"))]
    weather = pd.concat(frames, ignore_index=True)
    weather.to_parquet(OUT_FILE, index=False)

    # Also save the coordinates of the airports we have weather for.
    # The web app needs them to ask for a live forecast.
    coords.reset_index().loc[lambda d: d["iata"].isin(weather["origin"].unique())] \
        .to_parquet(DATA_DIR / "airports.parquet", index=False)
    print(f"\nSaved {len(weather):,} hourly weather rows for {weather['origin'].nunique()} airports")


if __name__ == "__main__":
    main()
