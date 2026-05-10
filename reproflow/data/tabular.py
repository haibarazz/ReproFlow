from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from reproflow.data.base import BaseDataAdapter, DataBundle
from reproflow.data.registry import get_data_config
from reproflow.data.schemas import as_list, normalize_task_type


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


class TabularDataset(Dataset):
    """Default Dataset for tabular/text-feature CSV tasks."""

    def __init__(
        self,
        df: pd.DataFrame,
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
            raise ValueError("TabularDataset requires at least one feature column.")
        missing = [col for col in [*self.feature_cols, self.label_col] if col not in self.df.columns]
        if missing:
            raise ValueError(f"Dataset is missing columns: {missing}")

        self.features = torch.tensor(self.df[self.feature_cols].values, dtype=torch.float32)
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


class TabularDataAdapter(BaseDataAdapter):
    """Default adapter that turns a CSV data config into train/test loaders."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.data_cfg = get_data_config(cfg)

    def build(self) -> DataBundle:
        df = self.load_raw()
        train_loader, test_loader, input_dim, meta = self.prepare(df)
        return DataBundle(train_loader=train_loader, test_loader=test_loader, input_dim=input_dim, meta=meta)

    def load_raw(self) -> pd.DataFrame:
        data_path = to_absolute_path(str(self.data_cfg.path))
        print(f"Loading data: {data_path}")
        df = pd.read_csv(data_path)
        self._require_columns(df)
        return df

    def prepare(self, df: pd.DataFrame):
        task_type = normalize_task_type(str(self.data_cfg.task_type))
        label_col = str(self.data_cfg.label_col)
        sample_id_col = str(self.data_cfg.get("id_col", "")) if self.data_cfg.get("id_col") else None

        train_df, test_df = self._split_dataframe(df, label_col, task_type)
        train_df, test_df, num_classes, label_meta = self._encode_labels(train_df, test_df, label_col, task_type)
        train_df, test_df, feature_cols, preprocessors = self._preprocess_features(train_df, test_df)
        preprocessors.update(label_meta)

        batch_size = int(self.cfg.training_loop.get("batch_size", 256))
        train_dataset = self._make_dataset(train_df, feature_cols, label_col, task_type, sample_id_col)
        test_dataset = self._make_dataset(test_df, feature_cols, label_col, task_type, sample_id_col)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(self.data_cfg.get("num_workers", 0)),
            pin_memory=bool(self.data_cfg.get("pin_memory", False)),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(self.data_cfg.get("num_workers", 0)),
            pin_memory=bool(self.data_cfg.get("pin_memory", False)),
        )
        data_meta = PreparedDataMeta(
            task_type=task_type,
            label_col=label_col,
            feature_cols=feature_cols,
            numeric_cols=as_list(self.data_cfg.get("numeric_cols")),
            categorical_cols=as_list(self.data_cfg.get("categorical_cols")),
            text_cols=as_list(self.data_cfg.get("text_cols")),
            num_classes=num_classes,
            train_size=len(train_dataset),
            test_size=len(test_dataset),
            preprocessors=preprocessors,
        ).as_dict()

        print(f"Prepared data: train={len(train_dataset)}, test={len(test_dataset)}, features={len(feature_cols)}")
        print(f"Task: {task_type}, label={label_col}, num_classes={num_classes}")
        return train_loader, test_loader, len(feature_cols), data_meta

    def _make_dataset(
        self,
        frame: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
        task_type: str,
        sample_id_col: str | None,
    ):
        dataset_cfg = self.data_cfg.get("dataset", {})
        if isinstance(dataset_cfg, DictConfig) and dataset_cfg.get("_target_"):
            return hydra.utils.instantiate(
                dataset_cfg,
                df=frame,
                feature_cols=feature_cols,
                label_col=label_col,
                task_type=task_type,
                sample_id_col=sample_id_col,
                _recursive_=False,
            )
        return TabularDataset(frame, feature_cols, label_col, task_type, sample_id_col)

    def _require_columns(self, df: pd.DataFrame) -> None:
        label_col = str(self.data_cfg.label_col)
        numeric_cols = as_list(self.data_cfg.get("numeric_cols"))
        categorical_cols = as_list(self.data_cfg.get("categorical_cols"))
        text_cols = as_list(self.data_cfg.get("text_cols"))
        id_col = str(self.data_cfg.get("id_col", "")) if self.data_cfg.get("id_col") else ""
        missing = [col for col in [id_col, label_col, *numeric_cols, *categorical_cols, *text_cols] if col and col not in df.columns]
        if missing:
            raise ValueError(f"Data file is missing required columns: {missing}")

    def _split_dataframe(self, df: pd.DataFrame, label_col: str, task_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_cfg = self.data_cfg.get("split", {})
        train_ratio = float(split_cfg.get("train_ratio", 0.8))
        random_state = int(split_cfg.get("random_state", 42))
        stratify = df[label_col] if task_type != "regression" and self._can_stratify(df[label_col]) else None
        train_df, test_df = train_test_split(
            df,
            test_size=1.0 - train_ratio,
            random_state=random_state,
            stratify=stratify,
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    @staticmethod
    def _can_stratify(labels: pd.Series) -> bool:
        counts = labels.value_counts(dropna=False)
        return len(counts) > 1 and counts.min() >= 2

    @staticmethod
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

    def _preprocess_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
        numeric_cols = as_list(self.data_cfg.get("numeric_cols"))
        categorical_cols = as_list(self.data_cfg.get("categorical_cols"))
        text_cols = as_list(self.data_cfg.get("text_cols"))
        preprocess_cfg = self.data_cfg.get("preprocess", {})
        text_cfg = self.data_cfg.get("text", {})
        preprocessors: dict[str, Any] = {}

        if bool(preprocess_cfg.get("scale_numeric", True)):
            train_df, test_df, scalers = self._scale_numeric(train_df, test_df, numeric_cols)
            preprocessors["numeric_scalers"] = scalers

        if bool(preprocess_cfg.get("encode_categorical", True)):
            train_df, test_df, cat_mappings = self._encode_categorical(train_df, test_df, categorical_cols)
            preprocessors["categorical_mappings"] = cat_mappings

        train_df, test_df, text_feature_cols, text_vectorizer = self._add_text_features(
            train_df,
            test_df,
            text_cols,
            int(text_cfg.get("max_features", 256)),
        )
        preprocessors["text_vectorizer"] = text_vectorizer

        feature_cols = [*numeric_cols, *categorical_cols, *text_feature_cols]
        if not feature_cols:
            raise ValueError("Data config must define at least one numeric, categorical, or text feature.")
        return train_df, test_df, feature_cols, preprocessors

    @staticmethod
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
            test_df[col] = test_df[col].fillna("__MISSING__").astype(str).map(mapping).fillna(-1).astype(float)
            mappings[col] = mapping
        return train_df, test_df, mappings

    @staticmethod
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

    @staticmethod
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
        return pd.concat([train_df, train_text_df], axis=1), pd.concat([test_df, test_text_df], axis=1), feature_cols, vectorizer
