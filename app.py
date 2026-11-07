"""
Step 4: The "Will my flight be late?" web page (Streamlit).

Run from the project folder:   streamlit run app.py

The user picks an airline, origin, destination, date and scheduled departure time
(and optionally a flight number). The app:
  1. builds the SAME features the model was trained on (features.py),
  2. gets weather for the origin airport (live forecast if the date is within
     ~16 days, otherwise the typical weather for that airport, month and hour),
  3. asks the trained LightGBM model for the probability of arriving 15+ min late.
"""

import json
import pickle
from datetime import date, datetime, time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import streamlit as st

import features as F

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
# Loading (cached so the model is read only once per browser session)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    booster = lgb.Booster(model_file=str(MODEL_DIR / "lgb_model.txt"))
    with open(MODEL_DIR / "app_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
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
        return None                      # no internet -> the app falls back to climatology
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
    # If the user did not give a flight number, use the carrier's average for the flight rate.
    if flight_num is None:
        row["flight_rate"] = row["carrier_rate"]

    # Fixed category lists, so codes match the trained model.
    for col in F.CATEGORICAL_FEATURES:
        row[col] = pd.Categorical(row[col], categories=artifacts["categories"][col])
    return row[F.FEATURES], weather_source


# ----------------------------------------------------------------------------
# The page
# ----------------------------------------------------------------------------
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


def risk_label(p: float) -> tuple[str, str]:
    if p < 0.15:
        return "Low risk", "#2e7d32"
    if p < 0.30:
        return "Moderate risk", "#f9a825"
    return "High risk", "#c62828"


def main() -> None:
    st.set_page_config(page_title="Will my flight be late?", page_icon="✈️")
    booster, artifacts, metrics = load_model()

    st.title("✈️ Will my flight be late?")
    st.write(
        "Estimate the chance that a US domestic flight **arrives 15+ minutes late**, "
        "based on the BTS on-time records for 2023–2024."
    )

    carriers = artifacts["categories"]["carrier"]
    origins = artifacts["categories"]["origin"]

    with st.form("flight"):
        col1, col2 = st.columns(2)
        carrier = col1.selectbox(
            "Airline", carriers,
            index=carriers.index("AA") if "AA" in carriers else 0,   # default: American Airlines
            format_func=lambda c: f"{AIRLINE_NAMES.get(c, c)} ({c})",
        )
        flight_num = col2.number_input("Flight number (optional)", min_value=0, max_value=9999, value=0,
                                       help="Leave at 0 if you do not know it.")
        col3, col4 = st.columns(2)
        origin = col3.selectbox("From (origin airport)", origins,
                                index=origins.index("ATL") if "ATL" in origins else 0)
        dest_options = [a for a in origins if a != origin]
        dest = col4.selectbox("To (destination airport)", dest_options,
                              index=dest_options.index("LAX") if "LAX" in dest_options else 0)
        col5, col6 = st.columns(2)
        flight_day = col5.date_input("Date", value=date.today(),
                                     min_value=date(2023, 1, 1), max_value=date(2030, 12, 31))
        dep_time = col6.time_input("Scheduled departure time (local)", value=time(9, 0), step=300)
        submitted = st.form_submit_button("Check my flight", type="primary")

    if not submitted:
        st.info("Fill in your flight and press **Check my flight**.")
        return

    if origin == dest:
        st.error("Origin and destination must be different airports.")
        return

    row, weather_source = build_input(
        carrier, origin, dest,
        flight_num if flight_num > 0 else None,
        flight_day, dep_time, artifacts,
    )
    probability = float(booster.predict(row, num_iteration=artifacts["best_iteration"])[0])
    label, color = risk_label(probability)

    st.subheader("Result")
    st.markdown(
        f"<div style='font-size:3rem;font-weight:700'>{probability:.0%}</div>"
        f"<div style='color:{color};font-weight:600'>{label}: chance of arriving 15+ min late</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(probability, 1.0))

    st.markdown("**Why this estimate?**")
    st.write(
        f"- Typical late rate for **{AIRLINE_NAMES.get(carrier, carrier)}**: "
        f"{row['carrier_rate'].iloc[0]:.0%}\n"
        f"- Typical late rate for the route **{origin} → {dest}**: {row['route_rate'].iloc[0]:.0%}\n"
        f"- Typical late rate leaving **{origin}** around {dep_time.hour:02d}:00: "
        f"{row['origin_hour_rate'].iloc[0]:.0%}\n"
        f"- Weather at {origin} ({weather_source}): {weather_summary(row)}"
    )

    st.caption(
        f"Model test accuracy (AUC on Jul–Dec 2024 flights it never saw): "
        f"**{metrics['scores']['model']['auc']:.2f}**, compared with "
        f"{metrics['scores']['baseline_route_rate']['auc']:.2f} for the route average alone. "
        "Delays are noisy: this is a probability, not a guarantee. Cancellations are not predicted."
    )


if __name__ == "__main__":
    main()
