from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


BINARY_METRICS = {
    "acc",
    "accuracy",
    "balanced_acc",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "auc",
    "roc_auc",
    "auc_pr",
    "pr_auc",
    "average_precision",
    "log_loss",
    "brier",
}

MULTICLASS_METRICS = {
    "acc",
    "accuracy",
    "balanced_acc",
    "balanced_accuracy",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
    "f1_macro",
    "f1_weighted",
    "auc_ovr",
    "auc_ovo",
    "log_loss",
}


def binary_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    metric_names: Iterable[str],
    threshold: float = 0.5,
) -> dict[str, float]:
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(int)
    preds = (scores >= threshold).astype(int)
    results: dict[str, float] = {}

    for metric in metric_names:
        if metric in {"acc", "accuracy"}:
            results[metric] = float(accuracy_score(labels, preds))
        elif metric in {"balanced_acc", "balanced_accuracy"}:
            results[metric] = float(balanced_accuracy_score(labels, preds))
        elif metric == "precision":
            results[metric] = float(precision_score(labels, preds, zero_division=0))
        elif metric == "recall":
            results[metric] = float(recall_score(labels, preds, zero_division=0))
        elif metric == "f1":
            results[metric] = float(f1_score(labels, preds, zero_division=0))
        elif metric == "mcc":
            results[metric] = float(matthews_corrcoef(labels, preds))
        elif metric in {"auc", "roc_auc"}:
            results[metric] = _safe_binary_auc(labels, scores)
        elif metric in {"auc_pr", "pr_auc", "average_precision"}:
            results[metric] = _safe_average_precision(labels, scores)
        elif metric == "log_loss":
            results[metric] = _safe_log_loss(labels, np.vstack([1.0 - scores, scores]).T)
        elif metric == "brier":
            results[metric] = float(brier_score_loss(labels, scores))
    return results


def multiclass_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    metric_names: Iterable[str],
) -> dict[str, float]:
    scores = np.asarray(scores)
    labels = np.asarray(labels).reshape(-1).astype(int)
    preds = np.argmax(scores, axis=1)
    results: dict[str, float] = {}

    for metric in metric_names:
        if metric in {"acc", "accuracy"}:
            results[metric] = float(accuracy_score(labels, preds))
        elif metric in {"balanced_acc", "balanced_accuracy"}:
            results[metric] = float(balanced_accuracy_score(labels, preds))
        elif metric == "precision_macro":
            results[metric] = float(precision_score(labels, preds, average="macro", zero_division=0))
        elif metric == "precision_weighted":
            results[metric] = float(precision_score(labels, preds, average="weighted", zero_division=0))
        elif metric == "recall_macro":
            results[metric] = float(recall_score(labels, preds, average="macro", zero_division=0))
        elif metric == "recall_weighted":
            results[metric] = float(recall_score(labels, preds, average="weighted", zero_division=0))
        elif metric == "f1_macro":
            results[metric] = float(f1_score(labels, preds, average="macro", zero_division=0))
        elif metric == "f1_weighted":
            results[metric] = float(f1_score(labels, preds, average="weighted", zero_division=0))
        elif metric == "auc_ovr":
            results[metric] = _safe_multiclass_auc(labels, scores, multi_class="ovr")
        elif metric == "auc_ovo":
            results[metric] = _safe_multiclass_auc(labels, scores, multi_class="ovo")
        elif metric == "log_loss":
            results[metric] = _safe_log_loss(labels, scores)
    return results


def _safe_binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return 0.5


def _safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(average_precision_score(labels, scores))
    except ValueError:
        return 0.5


def _safe_multiclass_auc(labels: np.ndarray, scores: np.ndarray, multi_class: str) -> float:
    try:
        return float(roc_auc_score(labels, scores, multi_class=multi_class, average="weighted"))
    except ValueError:
        return 0.5


def _safe_log_loss(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(log_loss(labels, scores))
    except ValueError:
        return 0.0
