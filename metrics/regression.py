from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


REGRESSION_METRICS = {
    "mse",
    "rmse",
    "mae",
    "mape",
    "median_ae",
    "r2",
    "explained_variance",
}


def regression_metrics(scores: np.ndarray, labels: np.ndarray, metric_names: Iterable[str]) -> dict[str, float]:
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    results: dict[str, float] = {}

    for metric in metric_names:
        if metric == "mse":
            results[metric] = float(mean_squared_error(labels, scores))
        elif metric == "rmse":
            results[metric] = float(np.sqrt(mean_squared_error(labels, scores)))
        elif metric == "mae":
            results[metric] = float(mean_absolute_error(labels, scores))
        elif metric == "mape":
            results[metric] = float(mean_absolute_percentage_error(labels, scores))
        elif metric == "median_ae":
            results[metric] = float(median_absolute_error(labels, scores))
        elif metric == "r2":
            results[metric] = _safe_r2(labels, scores)
        elif metric == "explained_variance":
            results[metric] = float(explained_variance_score(labels, scores))
    return results


def _safe_r2(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(r2_score(labels, scores))
    except ValueError:
        return 0.0
