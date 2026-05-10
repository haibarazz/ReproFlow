from __future__ import annotations

import numpy as np


RANKING_METRICS = {"hit_rate", "ndcg"}


def hit_rate_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if labels.size == 0:
        return 0.0
    top_idx = np.argsort(scores)[::-1][:k]
    return float(np.max(labels[top_idx]) > 0)


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if labels.size == 0:
        return 0.0
    top_idx = np.argsort(scores)[::-1][:k]
    gains = labels[top_idx]
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(labels)[::-1][:k]
    idcg = float(np.sum(ideal * discounts[: ideal.size]))
    return dcg / idcg if idcg > 0 else 0.0
