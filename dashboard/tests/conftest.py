import numpy as np
import pandas as pd
import pytest

from src.preprocessing import prepare_data


@pytest.fixture
def hourly():
    index = pd.date_range("2020-01-01", periods=480, freq="h", name="origin")
    values = 1 + .4*np.sin(np.arange(480)*2*np.pi/24) + .001*np.arange(480)
    return pd.DataFrame({"energy_kwh": values, "observed_minutes": 60}, index=index)


@pytest.fixture
def data(hourly):
    return prepare_data(hourly)
