"""
Step 3: Train the LightGBM model and evaluate it honestly.

Time-based split (NOT random), because in real life we predict the FUTURE:
    TRAIN       2023-01  ..  2024-03   (15 months)   model learns from these
    VALIDATION  2024-04  ..  2024-06   (3 months)    picks the number of trees
    TEST        2024-07  ..  2024-12   (6 months)    touched ONCE, for the final score

Outputs (in models/):
    lgb_model.txt           the trained LightGBM model
    app_meta.json, *.parquet  lookup tables + category lists the web app needs
    metrics.json            the scores, for the README / portfolio

Run AFTER download_data.py and download_weather.py:   python train.py
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

import features as F
from artifacts import save_artifacts

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

TRAIN_END = pd.Timestamp("2024-04-01")    # train: before this date
TEST_START = pd.Timestamp("2024-07-01")   # test: on/after this date


def load_and_prepare() -> pd.DataFrame:
    """Load the raw flights, clean them, and attach time and weather features."""
    print("Loading flights ...")
    raw = pd.read_parquet(DATA_DIR / "flights_raw.parquet")
    flights = F.clean_flights(raw)
    flights = F.add_time_features(flights)

    weather_file = DATA_DIR / "weather_hourly.parquet"
    if weather_file.exists():
        print("Joining hourly weather ...")
        weather = pd.read_parquet(weather_file)
        flights = F.add_weather(flights, weather)
    else:
        print("WARNING: no weather file found, training without weather")
        for col in F.WEATHER_FEATURES:
            flights[col] = np.nan

    flights = flights.reset_index(drop=True)
    print(f"Clean flights: {len(flights):,}  |  late rate: {flights[F.TARGET].mean():.3f}")
    return flights


def add_categories(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Make carrier/origin/dest pandas categories with a FIXED list of values.

    The app must use the same category list as training, so the model maps
    e.g. "AA" to the same internal code in both places.
    """
    df = df.copy()
    for col in F.CATEGORICAL_FEATURES:
        df[col] = pd.Categorical(df[col], categories=categories[col])
    return df


def evaluate(name: str, y_true: np.ndarray, p: np.ndarray) -> dict:
    """Standard classification scores for a probability prediction."""
    result = {
        "rows": int(len(y_true)),
        "late_rate": float(y_true.mean()),
        "auc": float(roc_auc_score(y_true, p)),
        "log_loss": float(log_loss(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "accuracy_at_0.5": float(accuracy_score(y_true, p >= 0.5)),
    }
    print(f"  {name:<28} AUC={result['auc']:.4f}  logloss={result['log_loss']:.4f}  "
          f"brier={result['brier']:.4f}")
    return result


def calibration_table(y_true: np.ndarray, p: np.ndarray) -> list[dict]:
    """Split predictions into 10 groups: predicted vs actual late rate in each group.

    If the model is well calibrated, a flight predicted at "30%" really is late ~30% of the time.
    """
    groups = pd.qcut(p, 10, labels=False, duplicates="drop")
    table = (pd.DataFrame({"p": p, "y": y_true, "g": groups})
               .groupby("g").agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size")))
    return [{"predicted": round(r.predicted, 3), "actual": round(r.actual, 3), "n": int(r.n)}
            for r in table.itertuples()]


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    flights = load_and_prepare()

    # ---- 1. Split by date --------------------------------------------------
    train_raw = flights[flights["flight_date"] < TRAIN_END]
    valid_raw = flights[(flights["flight_date"] >= TRAIN_END) & (flights["flight_date"] < TEST_START)]
    test_raw = flights[flights["flight_date"] >= TEST_START]
    print(f"Train {len(train_raw):,} | Valid {len(valid_raw):,} | Test {len(test_raw):,}")

    # ---- 2. Target encodings (leakage-safe) ---------------------------------
    # Train rows: out-of-fold encodings (each row never sees its own result).
    train = F.add_encodings_out_of_fold(train_raw)
    # Validation rows: encodings learned from the full training period.
    tables_train = F.fit_encodings(train_raw)
    valid = F.apply_encodings(valid_raw, tables_train)
    # Test rows: encodings learned from train + validation (everything before the test period).
    tables_history = F.fit_encodings(pd.concat([train_raw, valid_raw]))
    test = F.apply_encodings(test_raw, tables_history)

    # ---- 3. Fixed category lists (saved for the app) -----------------------
    categories = {col: sorted(flights[col].dropna().unique().tolist())
                  for col in F.CATEGORICAL_FEATURES}
    train = add_categories(train, categories)
    valid = add_categories(valid, categories)
    test = add_categories(test, categories)

    X_train, y_train = train[F.FEATURES], train[F.TARGET]
    X_valid, y_valid = valid[F.FEATURES], valid[F.TARGET]

    # ---- 4. Train LightGBM ------------------------------------------------
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.1,
        "num_leaves": 127,            # size of each tree (more = more detail, more risk)
        "min_data_in_leaf": 200,      # every leaf must cover >= 200 flights (prevents overfitting)
        "feature_fraction": 0.8,      # each tree sees a random 80% of the features
        "bagging_fraction": 0.8,      # each tree sees a random 80% of the rows
        "bagging_freq": 1,
        "verbose": -1,
    }
    dtrain = lgb.Dataset(X_train, y_train, categorical_feature=F.CATEGORICAL_FEATURES)
    dvalid = lgb.Dataset(X_valid, y_valid, reference=dtrain, categorical_feature=F.CATEGORICAL_FEATURES)

    print("Training LightGBM ...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)],
    )
    print(f"Best number of trees: {model.best_iteration}")

    # ---- 5. Final honest test ----------------------------------------------
    X_test, y_test = test[F.FEATURES], test[F.TARGET].to_numpy()
    p_test = model.predict(X_test, num_iteration=model.best_iteration)

    print("\nTEST SET RESULTS (Jul-Dec 2024, never seen during training):")
    scores = {"model": evaluate("LightGBM model", y_test, p_test)}
    # Simple baselines: how much better is the model than a single lookup table?
    scores["baseline_route_rate"] = evaluate("baseline: route late rate",
                                             y_test, test["route_rate"].to_numpy())
    scores["baseline_carrier_rate"] = evaluate("baseline: carrier late rate",
                                               y_test, test["carrier_rate"].to_numpy())

    # ---- 6. Feature importance (what the model relies on) ------------------
    gain = model.feature_importance(importance_type="gain")
    importance = (pd.Series(gain, index=F.FEATURES).sort_values(ascending=False)
                  / gain.sum()).round(4)
    print("\nTop features (share of total gain):")
    print(importance.head(10).to_string())

    # ---- 7. Artifacts for the web app -------------------------------------
    # Encodings from ALL data (2023 - 2024) so the app uses the most recent history.
    all_tables = F.fit_encodings(flights)
    # Weather climatology: average weather per airport / month / hour (fallback for future dates).
    climo = (flights.groupby(["origin", flights["flight_date"].dt.month, "dep_hour"])
                    [F.WEATHER_FEATURES].mean().reset_index()
                    .rename(columns={"flight_date": "month"}))

    # Typical distance and scheduled flight time for each route (the app only knows the route).
    route_info = (flights.assign(route=flights["origin"] + "-" + flights["dest"])
                         .groupby("route")[["distance", "crs_elapsed"]].median())

    model.save_model(str(MODEL_DIR / "lgb_model.txt"))
    save_artifacts(MODEL_DIR,
                   features=F.FEATURES,
                   categories=categories,
                   encodings=all_tables,
                   climatology=climo,
                   route_info=route_info,
                   best_iteration=model.best_iteration)

    metrics = {
        "split": {"train_end": str(TRAIN_END.date()), "test_start": str(TEST_START.date()),
                  "train_rows": len(train), "valid_rows": len(valid), "test_rows": len(test)},
        "scores": scores,
        "calibration_test": calibration_table(y_test, p_test),
        "top_features": importance.head(10).to_dict(),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model and metrics to {MODEL_DIR}")


if __name__ == "__main__":
    main()
