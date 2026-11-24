"""Historical one-step forecasts and optional recursive forecasts using past energy only."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import features_at_origins


def save_bundle(path, result, data, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not store any train/validation/test arrays inside an inference bundle.
    transform = {"feature_names": data.feature_names, "scaler": data.scaler,
                 "y_mean": data.y_mean, "y_std": data.y_std}
    joblib.dump({"model": result.model, "transform": transform, "config": result.config,
                 "metadata": metadata or {}, "schema": 1}, path)


def load_bundle(path):
    # joblib/pickle files execute code: load only artifacts you generated or trust.
    return joblib.load(path)


def predict_frame(bundle, frame):
    transform = bundle["transform"]
    X = frame[transform["feature_names"]].to_numpy(float)
    if not np.isfinite(X).all():
        raise ValueError("Insufficient valid historical observations for this origin")
    if transform["scaler"] is not None:
        X = transform["scaler"].transform(X)
    return bundle["model"].predict(X) * transform["y_std"] + transform["y_mean"]


def forecast(bundle, hourly, origin, horizon=1):
    if not 1 <= horizon <= 24:
        raise ValueError("Forecast horizon must be between 1 and 24")
    origin = pd.Timestamp(origin)
    if origin not in hourly.index:
        raise ValueError("Origin must be a historical hourly timestamp")
    # Truncate FIRST. Future observations can never enter recursive feature creation.
    history = hourly.loc[:origin, ["energy_kwh"]].copy()
    rows = []
    for step in range(horizon):
        current = origin + pd.Timedelta(hours=step)
        frame = features_at_origins(history).loc[[current]]
        estimate = float(predict_frame(bundle, frame)[0])
        endpoint = current + pd.Timedelta(hours=1)
        actual = hourly.energy_kwh.get(endpoint, np.nan)
        rows.append({"forecast_origin": origin, "target_time": endpoint,
                     "horizon": step + 1, "predicted_kwh": estimate, "actual_kwh": actual})
        # Leave predictions un-clipped, consistently with measured evaluation metrics.
        history.loc[endpoint, "energy_kwh"] = estimate
    return pd.DataFrame(rows)
