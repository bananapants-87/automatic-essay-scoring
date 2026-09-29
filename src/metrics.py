from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error


def rounded_clipped_predictions(y_pred: np.ndarray, score_min: float, score_max: float) -> np.ndarray:
    return np.clip(np.rint(y_pred), score_min, score_max).astype(int)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, score_min: float, score_max: float) -> dict[str, float]:
    rounded = rounded_clipped_predictions(y_pred, score_min, score_max)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true.astype(int), rounded, weights="quadratic")),
    }
