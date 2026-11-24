"""Every feature at origin t is measurable using hourly observations through t."""
import numpy as np
import pandas as pd

LAGS = (1, 3, 6, 12, 24)
WINDOWS = (3, 6, 12, 24)


def features_at_origins(hourly):
    if not isinstance(hourly.index, pd.DatetimeIndex) or not hourly.index.is_monotonic_increasing:
        raise ValueError("A sorted DatetimeIndex is required")
    if hourly.index.has_duplicates:
        raise ValueError("Duplicate hourly origins are not allowed")
    # Reindex before shifts: a missing clock hour must never become a one-hour lag.
    observed = hourly["energy_kwh"].asfreq("h")
    # Causal bounded forward fill is for INPUTS ONLY. Labels remain strictly observed.
    energy = observed.ffill(limit=6)
    x = pd.DataFrame({"energy_current": energy, "current_missing": observed.isna().astype(float)})
    for lag in LAGS:
        x[f"lag_{lag}"] = energy.shift(lag)
    for window in WINDOWS:
        x[f"rolling_mean_{window}"] = energy.rolling(window, min_periods=window).mean()
        x[f"rolling_std_{window}"] = energy.rolling(window, min_periods=window).std(ddof=0)
    # Calendar covariates describe the next hourly interval, whose left endpoint is t.
    idx = x.index
    x["hour"], x["day"], x["weekday"], x["month"] = idx.hour, idx.day, idx.dayofweek, idx.month
    x["weekend"] = (idx.dayofweek >= 5).astype(float)
    for name, values, period in (("hour", idx.hour, 24), ("weekday", idx.dayofweek, 7), ("month", idx.month - 1, 12)):
        x[name + "_sin"] = np.sin(2 * np.pi * values / period)
        x[name + "_cos"] = np.cos(2 * np.pi * values / period)
    return x.replace([np.inf, -np.inf], np.nan)


def supervised_frame(hourly):
    x = features_at_origins(hourly)
    x["target"] = hourly["energy_kwh"].asfreq("h").shift(-1)
    x["target_time"] = x.index + pd.Timedelta(hours=1)
    return x.dropna().copy()
