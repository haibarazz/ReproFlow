"""Example-only recommender data adapter and Dataset scaffolds."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from reproflow.data.base import BaseDataAdapter, DataBundle


class PairwiseRankingDataset(Dataset):
    """EXAMPLE ONLY: Dataset contract for pairwise recommender models."""

    def __init__(self, rows: list[dict[str, Any]], user_col: str = "user_id", pos_item_col: str = "pos_item_id", neg_item_col: str = "neg_item_id"):
        self.rows = rows
        self.user_col = user_col
        self.pos_item_col = pos_item_col
        self.neg_item_col = neg_item_col

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        return {
            "user_id": torch.tensor(int(row[self.user_col]), dtype=torch.long),
            "pos_item_id": torch.tensor(int(row[self.pos_item_col]), dtype=torch.long),
            "neg_item_id": torch.tensor(int(row[self.neg_item_col]), dtype=torch.long),
            "label": torch.tensor(1.0, dtype=torch.float32),
        }


class ImplicitFeedbackAdapter(BaseDataAdapter):
    """EXAMPLE ONLY: placeholder for user-item implicit feedback preparation."""

    def __init__(self, cfg):
        self.cfg = cfg

    def build(self) -> DataBundle:
        raise NotImplementedError(
            "ImplicitFeedbackAdapter is an example scaffold. Implement user/item ID mapping, "
            "negative sampling, split strategy, and PairwiseRankingDataset construction for a real task."
        )
