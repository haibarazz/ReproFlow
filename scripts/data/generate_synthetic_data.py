from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


NUM_COLS = [f"num_{i}" for i in range(5)]
CAT_COLS = [f"cat_{i}" for i in range(3)]


def _base_features(num_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    nums = rng.normal(0, 1, size=(num_samples, len(NUM_COLS)))
    cats = rng.integers(0, 4, size=(num_samples, len(CAT_COLS)))
    df = pd.DataFrame(nums, columns=NUM_COLS)
    for idx, col in enumerate(CAT_COLS):
        df[col] = cats[:, idx]
    df.insert(0, "sample_id", [f"sample_{seed}_{i:05d}" for i in range(num_samples)])
    return df


def make_binary(num_samples: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _base_features(num_samples, seed)
    signal = (
        1.6 * df["num_0"]
        - 1.1 * df["num_2"]
        + 0.7 * df["num_3"]
        + 0.4 * (df["cat_0"] == 1).astype(float)
        - 0.3 * (df["cat_2"] == 3).astype(float)
        + rng.normal(0, 0.7, size=num_samples)
    )
    df["label"] = (signal > np.median(signal)).astype(int)
    return df


def make_multiclass(num_samples: int = 900, seed: int = 43) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _base_features(num_samples, seed)
    score_a = df["num_0"] + 0.6 * df["num_1"] + rng.normal(0, 0.4, size=num_samples)
    score_b = -df["num_2"] + 0.8 * (df["cat_1"] == 2).astype(float) + rng.normal(0, 0.4, size=num_samples)
    score_c = df["num_3"] - 0.7 * df["num_4"] + 0.5 * (df["cat_2"] == 1).astype(float)
    logits = np.vstack([score_a, score_b, score_c]).T
    df["label"] = np.argmax(logits, axis=1)
    return df


def make_regression(num_samples: int = 800, seed: int = 44) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _base_features(num_samples, seed)
    df["label"] = (
        2.0 * df["num_0"]
        - 1.5 * df["num_1"]
        + 0.8 * df["num_4"]
        + 0.5 * df["cat_0"].astype(float)
        - 0.25 * df["cat_2"].astype(float)
        + rng.normal(0, 0.5, size=num_samples)
    )
    return df


def main() -> None:
    out_dir = Path("dataset")
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "sample_binary.csv": make_binary(),
        "sample_multiclass.csv": make_multiclass(),
        "sample_regression.csv": make_regression(),
    }
    for filename, df in datasets.items():
        path = out_dir / filename
        df.to_csv(path, index=False)
        print(f"Saved {path} with shape {df.shape}")

    datasets["sample_binary.csv"].to_csv(out_dir / "data.csv", index=False)


if __name__ == "__main__":
    main()
