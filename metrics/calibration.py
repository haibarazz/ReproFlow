from __future__ import annotations

import numpy as np


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    labels = np.asarray(labels).reshape(-1).astype(int)
    probabilities = np.asarray(probabilities).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (probabilities > lower) & (probabilities <= upper)
        if not np.any(mask):
            continue
        confidence = float(np.mean(probabilities[mask]))
        accuracy = float(np.mean(labels[mask] == (probabilities[mask] >= 0.5)))
        ece += float(np.mean(mask)) * abs(accuracy - confidence)
    return ece
