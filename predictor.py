"""
The prediction engine: turns one flight into a late-arrival probability.

Used by the chat (chatbot.py via app.py). It builds the SAME features the model
was trained on (features.py), looks up the weather, and asks LightGBM.
"""

import json
from datetime import date, datetime, time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import streamlit as st

import features as F
from artifacts import load_artifacts

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_DAYS = 16

# Friendly names for the most common airline codes (anything else shows its code).
AIRLINE_NAMES = {
    "AA": "American Airlines", "DL": "Delta Air Lines", "UA": "United Airlines",
    "WN": "Southwest Airlines", "AS": "Alaska Airlines", "B6": "JetBlue Airways",
    "NK": "Spirit Airlines", "F9": "Frontier Airlines", "HA": "Hawaiian Airlines",
    "G4": "Allegiant Air", "MQ": "Envoy Air", "OO": "SkyWest Airlines",
    "YX": "Republic Airways", "YV": "Mesa Airlines", "OH": "PSA Airlines",
    "9E": "Endeavor Air", "EV": "ExpressJet",
}


# ----------------------------------------------------------------------------
# Loading (cached so the model is read only once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    booster = lgb.Booster(model_file=str(MODEL_DIR / "lgb_model.txt"))
    artifacts = load_artifacts(MODEL_DIR)     # parquet + JSON, no pickle
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    return booster, artifacts, metrics


@st.cache_data
def load_airport_coordinates() -> dict:
    """IATA code -> (lat, lon), used for the live weather forecast."""
    table = pd.read_parquet(DATA_DIR / "airports.parquet")
    return {row.iata: (row.lat, row.lon) for row in table.itertuples()}


@st.cache_data(ttl=3600)
def fetch_forecast(lat: float, lon: float) -> pd.DataFrame | None:
    """Hourly weather forecast for the next ~16 days, in local airport time."""
    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "forecast_days": FORECAST_DAYS, "wind_speed_unit": "kmh",
        "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m",
    }
    try:
        response = requests.get(FORECAST_URL, params=params, timeout=15)
        response.raise_for_status()
    except requests.RequestException:
        return None                      # no internet -> fall back to typical weather
    hourly = response.json()["hourly"]
    return pd.DataFrame({
        "local_time": pd.to_datetime(hourly["time"]),
        "temperature_c": hourly["temperature_2m"],
        "precipitation_mm": hourly["precipitation"],
        "wind_kmh": hourly["wind_speed_10m"],
        "wind_gust_kmh": hourly["wind_gusts_10m"],
    })


# ----------------------------------------------------------------------------
# Building the model input for ONE flight
# ----------------------------------------------------------------------------
def weather_for(origin: str, flight_day: date, dep_hour: int, artifacts: dict) -> tuple[dict, str]:
    """Return the weather values and a note saying where they came from."""
    empty = {c: np.nan for c in F.WEATHER_FEATURES}
    days_ahead = (flight_day - date.today()).days

    # 1) Live forecast, if the flight is within the forecast window.
    coords = load_airport_coordinates().get(origin)
    if coords and 0 <= days_ahead < FORECAST_DAYS:
        forecast = fetch_forecast(*coords)
        if forecast is not None:
            target = pd.Timestamp(datetime.combine(flight_day, time(dep_hour)))
            row = forecast[forecast["local_time"] == target]
            if not row.empty:
                return row.iloc[0][F.WEATHER_FEATURES].to_dict(), "live forecast"

    # 2) Typical weather for this airport, month and hour (from 2023-2024 history).
    climo = artifacts["climatology"]
    row = climo[(climo["origin"] == origin) & (climo["month"] == flight_day.month)
                & (climo["dep_hour"] == dep_hour)]
    if not row.empty:
        return row.iloc[0][F.WEATHER_FEATURES].to_dict(), "typical weather (history)"

    return empty, "no weather data for this airport"


def build_input(carrier, origin, dest, flight_num, flight_day, dep_time, artifacts) -> tuple[pd.DataFrame, str]:
    """Create one row with exactly the columns the model was trained on."""
    dep_hour = dep_time.hour
    weather, weather_source = weather_for(origin, flight_day, dep_hour, artifacts)

    # Distance and flight time: typical values for this route.
    route = f"{origin}-{dest}"
    info = artifacts["route_info"]
    distance, elapsed = (info.loc[route, "distance"], info.loc[route, "crs_elapsed"]) \
        if route in info.index else (np.nan, np.nan)

    row = pd.DataFrame([{
        "carrier": carrier,
        "origin": origin,
        "dest": dest,
        "flight_num": flight_num,
        # BTS day of week: 1 = Monday ... 7 = Sunday (isoweekday)
        "month": flight_day.month,
        "day_of_week": flight_day.isoweekday(),
        "day_of_month": flight_day.day,
        "dep_minutes": dep_time.hour * 60 + dep_time.minute,
        "dep_hour": dep_hour,
        "distance": distance,
        "crs_elapsed": elapsed,
        **weather,
    }])

    # Historical late rates (the same lookup tables the training code used).
    row = F.apply_encodings(row, artifacts["encodings"])
    # Without a flight number, use the carrier's average for the flight rate.
    if flight_num is None:
        row["flight_rate"] = row["carrier_rate"]

    # Fixed category lists, so codes match the trained model.
    for col in F.CATEGORICAL_FEATURES:
        row[col] = pd.Categorical(row[col], categories=artifacts["categories"][col])
    return row[F.FEATURES], weather_source


def weather_summary(row: pd.DataFrame) -> str:
    """Human-readable weather values for the one flight, or a note if none were found."""
    values = row[F.WEATHER_FEATURES].iloc[0]
    if values.isna().all():
        return "no data available"

    def show(x, unit):
        return "n/a" if pd.isna(x) else f"{x:.0f} {unit}"

    return (f"{show(values['temperature_c'], '°C')}, "
            f"precipitation {show(values['precipitation_mm'], 'mm')}, "
            f"wind {show(values['wind_kmh'], 'km/h')} "
            f"(gusts {show(values['wind_gust_kmh'], 'km/h')})")


def risk_label(p: float) -> str:
    if p < 0.15:
        return "Low risk"
    if p < 0.30:
        return "Moderate risk"
    return "High risk"


def predict_flight(carrier, origin, dest, flight_num, flight_day, dep_time) -> dict:
    """Full prediction for one flight. Returns the numbers the chat needs to explain itself."""
    booster, artifacts, _metrics = load_model()
    row, weather_source = build_input(carrier, origin, dest, flight_num, flight_day, dep_time, artifacts)
    probability = float(booster.predict(row, num_iteration=artifacts["best_iteration"])[0])
    return {
        "probability": probability,
        "label": risk_label(probability),
        "carrier_rate": float(row["carrier_rate"].iloc[0]),
        "route_rate": float(row["route_rate"].iloc[0]),
        "origin_hour_rate": float(row["origin_hour_rate"].iloc[0]),
        "weather_source": weather_source,
        "weather_text": weather_summary(row),
    }
