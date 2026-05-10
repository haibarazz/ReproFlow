from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class DataBundle:
    """Prepared data returned by a ReproFlow data adapter."""

    train_loader: Any
    test_loader: Any
    input_dim: int
    meta: dict[str, Any]


class BaseDataAdapter(ABC):
    """Base class for data preparation plugins."""

    @abstractmethod
    def build(self) -> DataBundle:
        """Prepare train/test loaders plus metadata."""
