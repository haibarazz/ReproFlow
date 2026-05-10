from __future__ import annotations

from typing import Any

import numpy as np

from metrics import calculate_metrics, get_metric_names


class Evaluator:
    """Task-aware metric facade used by trainers and future offline evaluators."""

    def __init__(self, task_type: str, metrics_cfg: Any = None) -> None:
        self.task_type = task_type
        self.metrics_cfg = metrics_cfg or {}
        self.metric_names = get_metric_names(metrics_cfg, task_type)

    def evaluate(self, scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        return calculate_metrics(self.task_type, scores, labels, self.metrics_cfg)
