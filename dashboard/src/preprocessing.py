from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.feature_engineering import supervised_frame


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    frames: dict
    feature_names: list
    scaler: object
    y_mean: float
    y_std: float
    scaling: str

    def inverse_target(self, y):
        return np.asarray(y) * self.y_std + self.y_mean

    def transform(self, frame):
        x = frame[self.feature_names].to_numpy(dtype=float)
        return self.scaler.transform(x) if self.scaler is not None else x


def prepare_data(hourly, scaling="standard", max_hours=None):
    if scaling not in {"standard", "minmax", "robust", "none"}:
        raise ValueError("Unknown scaling")
    if max_hours is not None:
        if max_hours < 240:
            raise ValueError("Use at least 240 hours")
        hourly = hourly.iloc[:max_hours]
    frame = supervised_frame(hourly)
    if len(frame) < 100:
        raise ValueError("Insufficient complete causal feature/target rows")
    n = len(frame)
    a, b = int(n * .70), int(n * .85)
    train, val, test = frame.iloc[:a].copy(), frame.iloc[a:b].copy(), frame.iloc[b:].copy()
    # A training target cannot land at the first validation origin. Same for val/test.
    train = train.loc[train.target_time < val.index.min()]
    val = val.loc[val.target_time < test.index.min()]
    names = [c for c in frame if c not in {"target", "target_time"}]
    scaler = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}.get(scaling)
    scaler = scaler().fit(train[names].to_numpy(float)) if scaler else None
    transform = lambda f: scaler.transform(f[names].to_numpy(float)) if scaler else f[names].to_numpy(float)
    mean, std = float(train.target.mean()), float(train.target.std(ddof=0))
    std = max(std, 1e-8)
    target = lambda f: (f.target.to_numpy(float) - mean) / std
    return PreparedData(transform(train), transform(val), transform(test),
                        target(train), target(val), target(test),
                        {"train": train, "validation": val, "test": test}, names,
                        scaler, mean, std, scaling)
