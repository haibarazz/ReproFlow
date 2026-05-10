from __future__ import annotations

from typing import Any

from omegaconf import ListConfig


TASK_ALIASES = {
    "classification": "binary_classification",
    "binary": "binary_classification",
    "binary_classification": "binary_classification",
    "multiclass": "multiclass_classification",
    "Multiclass": "multiclass_classification",
    "multi_class": "multiclass_classification",
    "multiclass_classification": "multiclass_classification",
    "regression": "regression",
}


def normalize_task_type(task_type: str) -> str:
    normalized = TASK_ALIASES.get(str(task_type))
    if normalized is None:
        supported = ", ".join(sorted(set(TASK_ALIASES.values())))
        raise ValueError(f"Unsupported task_type={task_type!r}. Supported: {supported}")
    return normalized


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(item) for item in value]
    return [str(value)]
