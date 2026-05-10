from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Generic tabular dataset driven by configs/data/*.yaml."""

    def __init__(
        self,
        df,
        feature_cols: Sequence[str],
        label_col: str,
        task_type: str,
        sample_id_col: str | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.feature_cols = list(feature_cols)
        self.label_col = label_col
        self.task_type = task_type
        self.sample_id_col = sample_id_col if sample_id_col in self.df.columns else None

        if not self.feature_cols:
            raise ValueError("BaseDataset requires at least one feature column.")
        missing = [col for col in self.feature_cols + [self.label_col] if col not in self.df.columns]
        if missing:
            raise ValueError(f"Dataset is missing columns: {missing}")

        self.features = torch.tensor(
            self.df[self.feature_cols].values,
            dtype=torch.float32,
        )

        if self.task_type == "multiclass_classification":
            self.labels = torch.tensor(self.df[self.label_col].values, dtype=torch.long)
        else:
            self.labels = torch.tensor(self.df[self.label_col].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        item: dict[str, torch.Tensor | str] = {
            "basic_features": self.features[idx],
            "label": self.labels[idx],
        }
        if self.sample_id_col is not None:
            item["sample_id"] = str(self.df.iloc[idx][self.sample_id_col])
        return item
