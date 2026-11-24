import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted):
    if not np.isfinite(predicted).all():
        raise ValueError("Non-finite prediction")
    mse = mean_squared_error(actual, predicted)
    return {"mse": float(mse), "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(actual, predicted)), "r2": float(r2_score(actual, predicted))}
