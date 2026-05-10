from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from Data_pre import normalize_task_type
from metrics.classification import BINARY_METRICS, MULTICLASS_METRICS, binary_metrics, multiclass_metrics
from metrics.regression import REGRESSION_METRICS, regression_metrics


DEFAULT_METRICS = {
    "binary_classification": ["acc", "f1", "recall", "precision", "mcc", "auc", "auc_pr"],
    "multiclass_classification": ["acc", "f1_macro", "f1_weighted", "auc_ovr"],
    "regression": ["mse", "rmse", "mae", "r2"],
}

SUPPORTED_METRICS = {
    "binary_classification": BINARY_METRICS,
    "multiclass_classification": MULTICLASS_METRICS,
    "regression": REGRESSION_METRICS,
}


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _task_metrics_cfg(metrics_cfg: Any, task_type: str) -> Any:
    by_task = _cfg_get(metrics_cfg, "by_task", None)
    if by_task is not None and hasattr(by_task, "get"):
        return by_task.get(task_type, {})
    return metrics_cfg or {}


def get_metric_names(metrics_cfg: Any, task_type: str) -> list[str]:
    task_type = normalize_task_type(task_type)
    task_cfg = _task_metrics_cfg(metrics_cfg, task_type)
    include = _cfg_get(task_cfg, "include", None)
    if include:
        return [str(metric) for metric in include]
    return list(DEFAULT_METRICS[task_type])


def get_threshold(metrics_cfg: Any, task_type: str) -> float:
    task_cfg = _task_metrics_cfg(metrics_cfg, task_type)
    return float(_cfg_get(task_cfg, "threshold", _cfg_get(metrics_cfg, "threshold", 0.5)))


def validate_metrics(metrics_cfg: Any, task_type: str) -> list[str]:
    task_type = normalize_task_type(task_type)
    metric_names = get_metric_names(metrics_cfg, task_type)
    supported = SUPPORTED_METRICS[task_type]
    return [metric for metric in metric_names if metric not in supported]


def calculate_metrics(
    task_type: str,
    scores: np.ndarray,
    labels: np.ndarray,
    metrics_cfg: Any = None,
) -> dict[str, float]:
    task_type = normalize_task_type(task_type)
    unknown = validate_metrics(metrics_cfg, task_type)
    if unknown:
        supported = ", ".join(sorted(SUPPORTED_METRICS[task_type]))
        raise ValueError(f"Unsupported metrics for {task_type}: {unknown}. Supported: {supported}")

    metric_names = get_metric_names(metrics_cfg, task_type)
    if task_type == "binary_classification":
        return binary_metrics(scores, labels, metric_names, threshold=get_threshold(metrics_cfg, task_type))
    if task_type == "multiclass_classification":
        return multiclass_metrics(scores, labels, metric_names)
    return regression_metrics(scores, labels, metric_names)
