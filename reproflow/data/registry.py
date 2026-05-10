from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from reproflow.data.base import DataBundle


def get_data_config(cfg: DictConfig) -> DictConfig:
    if "data" in cfg:
        return cfg.data
    legacy = {
        "name": "legacy",
        "path": cfg.get("data_dir", "dataset/data.csv"),
        "task_type": cfg.get("training_loop", {}).get("task_type", "binary_classification"),
        "label_col": "label",
        "numeric_cols": [f"num_{idx}" for idx in range(5)],
        "categorical_cols": [f"cat_{idx}" for idx in range(3)],
        "text_cols": [],
        "split": {
            "strategy": "random",
            "train_ratio": cfg.get("training_loop", {}).get("train_ratio", 0.8),
            "random_state": cfg.get("training_loop", {}).get("random_state", 42),
        },
        "preprocess": {
            "scale_numeric": True,
            "encode_categorical": True,
        },
        "text": {
            "max_features": 256,
        },
    }
    return OmegaConf.create(legacy)


def _default_adapter_config() -> dict[str, Any]:
    return {"_target_": "reproflow.data.tabular.TabularDataAdapter"}


def _to_plain_config(config: Any) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        return dict(OmegaConf.to_container(config, resolve=False))
    return dict(config)


def _instantiate_adapter(adapter_cfg: Any, cfg: DictConfig):
    plain_cfg = _to_plain_config(adapter_cfg)
    target = plain_cfg.pop("_target_", None)
    if not target:
        raise ValueError("data.adapter must define _target_.")
    adapter_cls = hydra.utils.get_class(str(target))
    return adapter_cls(cfg=cfg, **plain_cfg)


def build_data_bundle(cfg: DictConfig) -> DataBundle:
    data_cfg = get_data_config(cfg)
    adapter_cfg = data_cfg.get("adapter") or _default_adapter_config()
    adapter = _instantiate_adapter(adapter_cfg, cfg)
    bundle = adapter.build()
    if not isinstance(bundle, DataBundle):
        raise TypeError("Data adapter build() must return reproflow.data.base.DataBundle.")
    return bundle
