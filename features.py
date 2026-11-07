"""
Shared feature engineering.

Training (train.py) and the web app (app.py) BOTH use this file, so a flight
is always turned into model inputs in exactly the same way. If the two code
paths disagreed, the model would silently get the wrong numbers.

Three kinds of features:
  1. Clean-up of the raw BTS rows (remove cancelled / diverted flights).
  2. Calendar and schedule features (month, weekday, departure time, route length).
  3. Weather at the origin airport at the scheduled departure hour.
  4. "Target encodings": the historical late-arrival rate of a carrier, route,
     airport or specific flight number. These are the most powerful features.
"""

import numpy as np
import pandas as pd

TARGET = "ArrDel15"          # 1 = arrived 15+ minutes late, 0 = on time

# Categorical columns: LightGBM treats them as categories, not numbers.
CATEGORICAL_FEATURES = ["carrier", "origin", "dest"]

# Historical late-rate features, filled in by the encoding step below.
RATE_FEATURES = ["carrier_rate", "origin_rate", "dest_rate",
                 "route_rate", "flight_rate", "origin_hour_rate"]

WEATHER_FEATURES = ["temperature_c", "precipitation_mm", "wind_kmh", "wind_gust_kmh"]

TIME_FEATURES = ["month", "day_of_week", "day_of_month", "dep_minutes", "distance", "crs_elapsed"]

# The exact list of columns the model sees, in this order.
FEATURES = CATEGORICAL_FEATURES + TIME_FEATURES + WEATHER_FEATURES + RATE_FEATURES


# ----------------------------------------------------------------------------
# 1. Clean-up
# ----------------------------------------------------------------------------
def clean_flights(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only flights that actually operated and have a known arrival outcome."""
    df = df.rename(columns={
        "Reporting_Airline": "carrier",
        "Flight_Number_Reporting_Airline": "flight_num",
        "Origin": "origin",
        "Dest": "dest",
        "CRSDepTime": "crs_dep",
        "DayOfWeek": "day_of_week",
        "DayofMonth": "day_of_month",
        "Month": "month",
        "Distance": "distance",
        "CRSElapsedTime": "crs_elapsed",
    })
    # Cancelled or diverted flights have no arrival delay -> cannot be used for training.
    df = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)]
    # Drop rows missing anything we need (some BTS rows have no flight number or schedule).
    df = df.dropna(subset=[TARGET, "crs_dep", "crs_elapsed", "flight_num",
                           "carrier", "origin", "dest"])

    df["flight_date"] = pd.to_datetime(df["FlightDate"])
    df["crs_dep"] = df["crs_dep"].astype(int)
    df["flight_num"] = df["flight_num"].astype(int)
    df[TARGET] = df[TARGET].astype(int)
    return df.drop(columns=["Cancelled", "Diverted", "FlightDate"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Calendar and schedule features
# ----------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the scheduled departure time (e.g. 1435 = 14:35) into minutes since midnight."""
    df = df.copy()
    hours = (df["crs_dep"] // 100) % 24          # "2400" (midnight) becomes 0
    minutes = df["crs_dep"] % 100
    df["dep_minutes"] = hours * 60 + minutes
    df["dep_hour"] = hours                       # helper, used for the weather join
    return df


# ----------------------------------------------------------------------------
# 3. Weather join (origin airport, scheduled departure hour, local time)
# ----------------------------------------------------------------------------
def add_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach hourly weather at the origin airport for the scheduled departure hour.

    `weather` has columns: origin, local_time (hourly), temperature_c, ...
    Flights from airports without weather data keep NaN; LightGBM handles missing values.
    """
    df = df.copy()
    # Scheduled departure as a full timestamp, rounded down to the hour.
    df["dep_hour_ts"] = df["flight_date"] + pd.to_timedelta(df["dep_hour"], unit="h")

    weather = weather.rename(columns={"local_time": "dep_hour_ts"})
    df = df.merge(weather, on=["origin", "dep_hour_ts"], how="left")
    return df.drop(columns=["dep_hour_ts"])


# ----------------------------------------------------------------------------
# 4. Target encodings ("historical late rates")
# ----------------------------------------------------------------------------
# Each entry: (feature name, function that builds the grouping key from a DataFrame)
ENCODING_KEYS = {
    "carrier_rate":     lambda d: d["carrier"],
    "origin_rate":      lambda d: d["origin"],
    "dest_rate":        lambda d: d["dest"],
    "route_rate":       lambda d: d["origin"] + "-" + d["dest"],
    "flight_rate":      lambda d: d["carrier"] + d["flight_num"].astype(str),
    "origin_hour_rate": lambda d: d["origin"] + "@" + d["dep_hour"].astype(str),
}
SMOOTHING = 20   # a key seen fewer than ~20 times is pulled toward the overall average


def fit_encodings(df: pd.DataFrame) -> dict:
    """Learn the average late rate for every key, with smoothing for rare keys.

    Smoothing formula:  rate = (sum_late + m * overall) / (n + m)
    Rare keys (few flights) stay close to the overall average instead of
    jumping to 0% or 100% from a handful of flights.
    """
    overall = df[TARGET].mean()
    tables = {"overall": overall}
    for name, make_key in ENCODING_KEYS.items():
        keys = make_key(df)
        stats = df.groupby(keys)[TARGET].agg(["sum", "count"])
        tables[name] = (stats["sum"] + SMOOTHING * overall) / (stats["count"] + SMOOTHING)
    return tables


def apply_encodings(df: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """Look up each flight's encoded rates. Unknown keys get the overall average."""
    df = df.copy()
    for name, make_key in ENCODING_KEYS.items():
        df[name] = make_key(df).map(tables[name]).fillna(tables["overall"]).astype(float)
    return df


def add_encodings_out_of_fold(train: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Encode TRAINING rows without letting each row see its own answer.

    If a flight's own late/on-time result were used to compute its own rate,
    the model would cheat during training and fail on new data. So we split
    the training rows into folds: each fold is encoded using the OTHER folds only.
    """
    rng = np.random.default_rng(seed)
    fold_id = rng.integers(0, n_folds, size=len(train))
    pieces = []
    for k in range(n_folds):
        in_fold = train[fold_id == k]
        out_fold = train[fold_id != k]
        tables = fit_encodings(out_fold)
        pieces.append(apply_encodings(in_fold, tables))
    return pd.concat(pieces).sort_index()
