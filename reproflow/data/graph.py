"""Example-only graph data adapter and Dataset scaffolds."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from reproflow.data.base import BaseDataAdapter, DataBundle


class GraphMiniBatchDataset(Dataset):
    """EXAMPLE ONLY: Dataset contract for graph-style model batches."""

    def __init__(self, graphs: list[dict[str, Any]]):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        graph = self.graphs[idx]
        return {
            "node_features": torch.as_tensor(graph["node_features"], dtype=torch.float32),
            "edge_index": torch.as_tensor(graph["edge_index"], dtype=torch.long),
            "label": torch.as_tensor(graph["label"], dtype=torch.float32),
        }


class GraphDataAdapter(BaseDataAdapter):
    """EXAMPLE ONLY: placeholder for graph construction and batching."""

    def __init__(self, cfg):
        self.cfg = cfg

    def build(self) -> DataBundle:
        raise NotImplementedError(
            "GraphDataAdapter is an example scaffold. Implement node/edge construction, graph split, "
            "and GraphMiniBatchDataset construction for a real task."
        )
