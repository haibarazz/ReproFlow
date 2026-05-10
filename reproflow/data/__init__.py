"""Data adapter plugin layer for ReproFlow.

New data shapes should be added here instead of growing Data_pre.py or
Dataset.py into large conditional files.
"""

from reproflow.data.base import BaseDataAdapter, DataBundle
from reproflow.data.registry import build_data_bundle
from reproflow.data.schemas import normalize_task_type
from reproflow.data.tabular import TabularDataAdapter, TabularDataset

__all__ = [
    "BaseDataAdapter",
    "DataBundle",
    "TabularDataAdapter",
    "TabularDataset",
    "build_data_bundle",
    "normalize_task_type",
]
