from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader

from Dataset import BaseDataset


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


@dataclass
class PreparedDataMeta:
    task_type: str
    label_col: str
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    text_cols: list[str]
    num_classes: int
    train_size: int
    test_size: int
    preprocessors: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "label_col": self.label_col,
            "feature_cols": self.feature_cols,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "text_cols": self.text_cols,
            "num_classes": self.num_classes,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "preprocessors": self.preprocessors,
        }


def normalize_task_type(task_type: str) -> str:
    normalized = TASK_ALIASES.get(str(task_type))
    if normalized is None:
        supported = ", ".join(sorted(set(TASK_ALIASES.values())))
        raise ValueError(f"Unsupported task_type={task_type!r}. Supported: {supported}")
    return normalized


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(item) for item in value]
    return [str(value)]


def get_data_config(cfg: DictConfig) -> DictConfig:
    if "data" in cfg:
        return cfg.data
    legacy = {
        "name": "legacy",
        "path": cfg.get("data_dir", "dataset/data.csv"),
        "task_type": cfg.get("training_loop", {}).get("task_type", "binary_classification"),
        "label_col": "label",
        "numeric_cols": [f"num_{i}" for i in range(5)],
        "categorical_cols": [f"cat_{i}" for i in range(3)],
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


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [col for col in cols if col and col not in df.columns]
    if missing:
        raise ValueError(f"Data file is missing required columns: {missing}")


def load_and_preprocess_data(cfg: DictConfig) -> pd.DataFrame:
    data_cfg = get_data_config(cfg)
    data_path = to_absolute_path(str(data_cfg.path))
    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)

    label_col = str(data_cfg.label_col)
    numeric_cols = _as_list(data_cfg.get("numeric_cols"))
    categorical_cols = _as_list(data_cfg.get("categorical_cols"))
    text_cols = _as_list(data_cfg.get("text_cols"))
    id_col = str(data_cfg.get("id_col", "")) if data_cfg.get("id_col") else ""
    _require_columns(df, [id_col, label_col, *numeric_cols, *categorical_cols, *text_cols])
    return df


def _can_stratify(labels: pd.Series) -> bool:
    counts = labels.value_counts(dropna=False)
    return len(counts) > 1 and counts.min() >= 2


def _split_dataframe(df: pd.DataFrame, cfg: DictConfig, label_col: str, task_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_cfg = cfg.get("split", {})
    train_ratio = float(split_cfg.get("train_ratio", 0.8))
    random_state = int(split_cfg.get("random_state", 42))
    stratify = df[label_col] if task_type != "regression" and _can_stratify(df[label_col]) else None

    train_df, test_df = train_test_split(
        df,
        test_size=1.0 - train_ratio,
        random_state=random_state,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _encode_labels(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    task_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, int, dict[str, Any]]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    meta: dict[str, Any] = {}

    if task_type == "regression":
        train_df[label_col] = pd.to_numeric(train_df[label_col], errors="raise").astype(float)
        test_df[label_col] = pd.to_numeric(test_df[label_col], errors="raise").astype(float)
        return train_df, test_df, 1, meta

    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([train_df[label_col], test_df[label_col]], axis=0).astype(str))
    train_df[label_col] = label_encoder.transform(train_df[label_col].astype(str))
    test_df[label_col] = label_encoder.transform(test_df[label_col].astype(str))
    num_classes = len(label_encoder.classes_)
    meta["label_classes"] = list(label_encoder.classes_)

    if task_type == "binary_classification" and num_classes != 2:
        raise ValueError(
            f"binary_classification requires exactly 2 label classes, got {num_classes}: "
            f"{list(label_encoder.classes_)}"
        )
    if task_type == "multiclass_classification" and num_classes < 3:
        raise ValueError("multiclass_classification requires at least 3 label classes.")
    return train_df, test_df, num_classes, meta


def _encode_categorical(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
    mappings: dict[str, dict[str, int]] = {}
    for col in categorical_cols:
        train_values = train_df[col].fillna("__MISSING__").astype(str)
        categories = sorted(train_values.unique())
        mapping = {value: idx for idx, value in enumerate(categories)}
        train_df[col] = train_values.map(mapping).astype(float)
        test_df[col] = (
            test_df[col]
            .fillna("__MISSING__")
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype(float)
        )
        mappings[col] = mapping
    return train_df, test_df, mappings


def _scale_numeric(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, StandardScaler]]:
    scalers: dict[str, StandardScaler] = {}
    for col in numeric_cols:
        train_values = pd.to_numeric(train_df[col], errors="coerce")
        median_value = train_values.median()
        if np.isnan(median_value):
            median_value = 0.0
        train_df[col] = train_values.fillna(median_value)
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce").fillna(median_value)

        scaler = StandardScaler()
        train_df[col] = scaler.fit_transform(train_df[[col]]).reshape(-1)
        test_df[col] = scaler.transform(test_df[[col]]).reshape(-1)
        scalers[col] = scaler
    return train_df, test_df, scalers


def _add_text_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_cols: list[str],
    max_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], TfidfVectorizer | None]:
    if not text_cols:
        return train_df, test_df, [], None

    def join_text(frame: pd.DataFrame) -> pd.Series:
        return frame[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

    vectorizer = TfidfVectorizer(max_features=max_features)
    train_matrix = vectorizer.fit_transform(join_text(train_df)).toarray()
    test_matrix = vectorizer.transform(join_text(test_df)).toarray()
    feature_cols = [f"tfidf_{idx:04d}" for idx in range(train_matrix.shape[1])]

    train_text_df = pd.DataFrame(train_matrix, columns=feature_cols, index=train_df.index)
    test_text_df = pd.DataFrame(test_matrix, columns=feature_cols, index=test_df.index)
    train_df = pd.concat([train_df, train_text_df], axis=1)
    test_df = pd.concat([test_df, test_text_df], axis=1)
    return train_df, test_df, feature_cols, vectorizer


def _preprocess_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    numeric_cols = _as_list(cfg.get("numeric_cols"))
    categorical_cols = _as_list(cfg.get("categorical_cols"))
    text_cols = _as_list(cfg.get("text_cols"))
    preprocess_cfg = cfg.get("preprocess", {})
    text_cfg = cfg.get("text", {})
    preprocessors: dict[str, Any] = {}

    if bool(preprocess_cfg.get("scale_numeric", True)):
        train_df, test_df, scalers = _scale_numeric(train_df, test_df, numeric_cols)
        preprocessors["numeric_scalers"] = scalers

    if bool(preprocess_cfg.get("encode_categorical", True)):
        train_df, test_df, cat_mappings = _encode_categorical(train_df, test_df, categorical_cols)
        preprocessors["categorical_mappings"] = cat_mappings

    max_text_features = int(text_cfg.get("max_features", 256))
    train_df, test_df, text_feature_cols, text_vectorizer = _add_text_features(
        train_df,
        test_df,
        text_cols,
        max_text_features,
    )
    preprocessors["text_vectorizer"] = text_vectorizer

    feature_cols = [*numeric_cols, *categorical_cols, *text_feature_cols]
    if not feature_cols:
        raise ValueError("Data config must define at least one numeric, categorical, or text feature.")
    return train_df, test_df, feature_cols, preprocessors


def prepare_data(df: pd.DataFrame, cfg: DictConfig):
    data_cfg = get_data_config(cfg)
    task_type = normalize_task_type(str(data_cfg.task_type))
    label_col = str(data_cfg.label_col)
    sample_id_col = str(data_cfg.get("id_col", "")) if data_cfg.get("id_col") else None

    train_df, test_df = _split_dataframe(df, data_cfg, label_col, task_type)
    train_df, test_df, num_classes, label_meta = _encode_labels(train_df, test_df, label_col, task_type)
    train_df, test_df, feature_cols, preprocessors = _preprocess_features(train_df, test_df, data_cfg)
    preprocessors.update(label_meta)

    batch_size = int(cfg.training_loop.get("batch_size", 256))
    train_dataset = BaseDataset(train_df, feature_cols, label_col, task_type, sample_id_col)
    test_dataset = BaseDataset(test_df, feature_cols, label_col, task_type, sample_id_col)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    data_meta = PreparedDataMeta(
        task_type=task_type,
        label_col=label_col,
        feature_cols=feature_cols,
        numeric_cols=_as_list(data_cfg.get("numeric_cols")),
        categorical_cols=_as_list(data_cfg.get("categorical_cols")),
        text_cols=_as_list(data_cfg.get("text_cols")),
        num_classes=num_classes,
        train_size=len(train_dataset),
        test_size=len(test_dataset),
        preprocessors=preprocessors,
    ).as_dict()

    print(f"Prepared data: train={len(train_dataset)}, test={len(test_dataset)}, features={len(feature_cols)}")
    print(f"Task: {task_type}, label={label_col}, num_classes={num_classes}")
    return train_loader, test_loader, len(feature_cols), data_meta
