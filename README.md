# Flight Delay Predictor

Estimate the chance that a US domestic flight arrives **15+ minutes late**.

The project trains a LightGBM model on about 13.7 million US domestic flights from 2023 and 2024, using the US Bureau of Transportation Statistics (BTS) on-time records. A Streamlit web page lets you enter an airline, route, date, and scheduled departure time, and returns a probability along with the weather used for the estimate.

## Results

Evaluated on flights from July to December 2024, which the model never saw during training:

| Model | AUC | Log loss | Brier score |
|---|---|---|---|
| LightGBM (this project) | **0.679** | **0.462** | **0.147** |
| Route average only | 0.562 | 0.494 | 0.158 |
| Carrier average only | 0.543 | 0.496 | 0.158 |

AUC measures how well the model ranks late flights above on-time ones (0.5 is random, 1.0 is perfect). The model is well above the simple averages, but delays are noisy, so its output is a probability, not a guarantee. Calibration by decile is saved in `models/metrics.json`.

The most useful features were the historical late rate of the origin airport at that hour, the history of the specific flight number, the month, the scheduled departure time, and the precipitation at the origin.

## Project structure

| File | Purpose |
|---|---|
| `download_data.py` | Downloads the monthly BTS files for 2023–2024 and saves the needed columns |
| `download_weather.py` | Downloads hourly weather for the 150 busiest airports (Open-Meteo) |
| `features.py` | Feature engineering, shared by training and the app |
| `train.py` | Time-based split, LightGBM training, evaluation, and saving the model |
| `artifacts.py` | Saves and loads the lookup tables as JSON and parquet (no pickle) |
| `app.py` | The Streamlit web page |
| `models/` | Trained model, lookup tables, and metrics |

`data/` is not in the repository. It is created by the scripts below.

## Setup

Requires Python 3.10 or newer.

```bash
pip install pandas numpy lightgbm scikit-learn streamlit requests pyarrow
```

## Usage

Run these once, in order, from the project folder:

```bash
python download_data.py
python download_weather.py
python train.py
```

`download_data.py` downloads about 650 MB from BTS. `download_weather.py` is rate-limited by the free weather API, so it may pause; re-running it resumes where it stopped. `train.py` takes a while on a normal computer.

To start the web page:

```bash
python -m streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## How it works

- **Data cleaning:** cancelled and diverted flights are removed, since they have no arrival outcome.
- **Target:** `ArrDel15`, whether the flight arrived 15 or more minutes late.
- **No leakage:** the actual departure and arrival delays are dropped as features, because they are only known after the flight.
- **Time-based split:** training uses 2023 through March 2024, validation uses April to June 2024 (for choosing the number of trees), and the final test uses July to December 2024.
- **Historical rates:** carrier, origin, destination, route, flight number, and origin-hour late rates are computed with smoothing. Training rows get out-of-fold rates, so no row sees its own result.
- **Weather:** training uses the actual weather at the origin airport in the scheduled departure hour. The app uses the live forecast for dates within 16 days, and typical weather for that airport, month, and hour for later dates.

## Limitations

- The model predicts late arrivals only. It does not predict cancellations.
- Weather for later dates is a typical value, not a forecast, so those estimates are less precise.
- Only the 150 busiest airports have weather data. Flights from other airports still get a prediction, but without weather.
- Results depend on 2023–2024 patterns. Airline schedules and airport operations change over time.

## Data source

On-time performance data: [US Bureau of Transportation Statistics](https://www.transtats.bts.gov/), Reporting Carrier On-Time Performance.
Weather: [Open-Meteo](https://open-meteo.com/) historical and forecast APIs.
Airport coordinates: [OpenFlights](https://openflights.org/data.html).
