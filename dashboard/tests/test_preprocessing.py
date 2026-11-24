import numpy as np
import pandas as pd
import pytest

from src.data_loader import aggregate_minutes
from src.feature_engineering import features_at_origins, supervised_frame
from src.preprocessing import prepare_data


def test_kw_to_kwh_and_partial_hours():
    t = pd.date_range("2020-01-01", periods=90, freq="min")
    frame = pd.DataFrame({"Date": t.strftime("%d/%m/%Y"), "Time": t.strftime("%H:%M:%S"), "Global_active_power": 2.})
    hourly, _ = aggregate_minutes(frame)
    assert hourly.index[0] == pd.Timestamp("2020-01-01 01:00")
    assert hourly.energy_kwh.iloc[0] == 2.
    assert np.isnan(hourly.energy_kwh.iloc[1])


def test_invalid_values_and_duplicate_minutes():
    t = pd.date_range("2020-01-01", periods=60, freq="min")
    frame = pd.DataFrame({"Date": t.strftime("%d/%m/%Y"), "Time": t.strftime("%H:%M:%S"), "Global_active_power": "1"})
    frame.loc[2, "Global_active_power"] = "?"
    frame.loc[3, "Global_active_power"] = "-1"
    frame = pd.concat([frame, frame.iloc[[0]]])
    hourly, stats = aggregate_minutes(frame)
    assert stats["duplicate_timestamps"] == 1
    assert hourly.observed_minutes.iloc[0] == 58
    assert hourly.energy_kwh.isna().all()


def test_features_ignore_future_mutations(hourly):
    origin = hourly.index[150]
    before = features_at_origins(hourly).loc[:origin]
    modified = hourly.copy()
    modified.loc[modified.index > origin, "energy_kwh"] = 999.
    pd.testing.assert_frame_equal(before, features_at_origins(modified).loc[:origin])
    frame = supervised_frame(hourly)
    assert frame.loc[origin, "target"] == hourly.energy_kwh.iloc[151]
    assert frame.loc[origin, "lag_24"] == hourly.energy_kwh.iloc[126]


def test_missing_targets_never_imputed(hourly):
    hourly.loc[hourly.index[100], "energy_kwh"] = np.nan
    assert hourly.index[99] not in supervised_frame(hourly).index
    assert features_at_origins(hourly).loc[hourly.index[100], "current_missing"] == 1


def test_scaler_fits_train_only_and_boundary_purge(hourly):
    data = prepare_data(hourly)
    names = data.feature_names
    np.testing.assert_allclose(data.scaler.mean_, data.frames["train"][names].mean(), atol=1e-12)
    assert data.frames["train"].target_time.max() < data.frames["validation"].index.min()
    assert data.frames["validation"].target_time.max() < data.frames["test"].index.min()
    modified = hourly.copy()
    modified.loc[modified.index >= data.frames["test"].index.min(), "energy_kwh"] += 1000
    changed = prepare_data(modified)
    np.testing.assert_allclose(data.X_train, changed.X_train)
    np.testing.assert_allclose(data.scaler.mean_, changed.scaler.mean_)


@pytest.mark.parametrize("scaling", ["standard", "minmax", "robust", "none"])
def test_scaling_is_finite(hourly, scaling):
    data = prepare_data(hourly, scaling)
    assert np.isfinite(data.X_train).all()
    np.testing.assert_allclose(data.inverse_target(data.y_test), data.frames["test"].target)
