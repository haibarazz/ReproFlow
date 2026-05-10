from __future__ import annotations

import time
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from tqdm import tqdm

from Data_pre import normalize_task_type
from evaluators import Evaluator
from metrics.registry import get_threshold
from reproflow.reports import generate_markdown_report
from reproflow.tracking import write_run_metadata


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class AbstractTrainer(ABC):
    @abstractmethod
    def train_epoch(self, train_loader):
        pass

    @abstractmethod
    def evaluate_epoch(self, test_loader):
        pass

    @abstractmethod
    def train(self, train_loader, test_loader):
        pass

    @abstractmethod
    def train_from_epoch(self, train_loader, test_loader, start_epoch=0):
        pass


class EarlyStopper:
    def __init__(self, patience: int, mode: str = "min", delta: float = 1e-8) -> None:
        self.patience = int(patience)
        self.mode = mode
        self.delta = float(delta)
        self.counter = 0
        self.best_value: float | None = None

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "max":
            return value > self.best_value + self.delta
        return value < self.best_value - self.delta

    def step(self, value: float) -> bool:
        if self._is_better(value):
            self.best_value = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class BaseTrainer(AbstractTrainer):
    expected_task_type: str | None = None

    def __init__(self, model, cfg, data_meta: dict[str, Any] | None = None, run_context: dict[str, Any] | None = None):
        self.model = model
        self.cfg = cfg
        self.data_meta = data_meta or {}
        self.run_context = run_context or {}
        self.task_type = normalize_task_type(
            self.data_meta.get("task_type", cfg.get("data", {}).get("task_type", "binary_classification"))
        )
        if self.expected_task_type and self.task_type != self.expected_task_type:
            raise ValueError(
                f"{self.__class__.__name__} expects task_type={self.expected_task_type}, "
                f"but data config uses {self.task_type}."
            )

        use_cuda = str(cfg.get("device", "cuda")).lower() == "cuda" and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(self.device)

        train_cfg = cfg.training_loop
        self.lr = float(train_cfg.get("learning_rate", 1e-3))
        self.epochs = int(train_cfg.get("epochs", 20))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-5))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        self.monitor_metric = str(train_cfg.get("monitor_metric", "val_loss"))
        self.monitor_mode = str(train_cfg.get("monitor_mode", "min")).lower()
        if self.monitor_mode not in {"min", "max"}:
            raise ValueError("training_loop.monitor_mode must be 'min' or 'max'.")

        self.evaluator = Evaluator(self.task_type, cfg.get("metrics", {}))
        self.best_model_path = Path(str(cfg.best_model_path))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = self._build_criterion()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self.monitor_mode,
            factor=float(train_cfg.get("scheduler_factor", 0.5)),
            patience=int(train_cfg.get("scheduler_patience", 10)),
        )
        self.early_stopper = EarlyStopper(
            patience=int(train_cfg.get("early_stop_patience", 20)),
            mode=self.monitor_mode,
            delta=float(train_cfg.get("early_stop_delta", 1e-8)),
        )

        self.training_history: list[dict[str, Any]] = []
        self.setup_logging()

    def setup_logging(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(str(self.cfg.log_path))
        log_dir.mkdir(parents=True, exist_ok=True)
        artifact_cfg = self.cfg.get("artifacts", {})
        self.save_events = bool(artifact_cfg.get("save_events", False))
        self.save_metrics_json = bool(artifact_cfg.get("save_metrics_json", False))
        self.save_predictions = bool(artifact_cfg.get("save_predictions", False))
        self.save_manifest = bool(artifact_cfg.get("save_manifest", False))
        self.save_structured_artifacts = any(
            [self.save_events, self.save_metrics_json, self.save_predictions, self.save_manifest]
        )
        self.run_artifact_dir = Path(str(self.run_context.get("run_dir") or log_dir)) if self.save_structured_artifacts else log_dir
        self.run_artifact_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = log_dir / f"training_{timestamp}.log"
        self.history_file = log_dir / f"history_{timestamp}.csv"
        self.config_snapshot_file = log_dir / f"config_{timestamp}.yaml"
        self.events_file = self.run_artifact_dir / "events.jsonl"
        self.metrics_file = self.run_artifact_dir / "metrics_latest.json"
        self.predictions_file = self.run_artifact_dir / "predictions.csv"
        self.manifest_file = self.run_artifact_dir / "artifacts_manifest.json"
        OmegaConf.save(config=self.cfg, f=self.config_snapshot_file, resolve=True)

        self.log_and_print(self.format_config())
        self.log_and_print("=" * 80)
        self.log_event(
            "run_started",
            {
                "log_file": self.log_file,
                "history_file": self.history_file,
                "config_snapshot_file": self.config_snapshot_file,
                "run_artifact_dir": self.run_artifact_dir,
            },
        )

    def format_config(self) -> str:
        model_name = getattr(self.model, "name", self.model.__class__.__name__)
        return (
            "Experiment config\n"
            f"model: {model_name}\n"
            f"task_type: {self.task_type}\n"
            f"epochs: {self.epochs}\n"
            f"learning_rate: {self.lr}\n"
            f"weight_decay: {self.weight_decay}\n"
            f"monitor: {self.monitor_metric} ({self.monitor_mode})\n"
            f"device: {self.device}\n"
            f"feature_count: {len(self.data_meta.get('feature_cols', []))}\n"
            f"experiment_id: {self.run_context.get('experiment_id', 'n/a')}\n"
            f"run_id: {self.run_context.get('run_id', 'n/a')}\n"
            f"started_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"config_snapshot: {self.config_snapshot_file}"
        )

    def log_and_print(self, message: str) -> None:
        print(message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")

    def log_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        if not self.save_events:
            return
        event = {
            "event": event_type,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "experiment_id": self.run_context.get("experiment_id", "n/a"),
            "run_id": self.run_context.get("run_id", "n/a"),
            **(payload or {}),
        }
        with open(self.events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(event), ensure_ascii=False, default=str) + "\n")

    def _build_criterion(self):
        if self.task_type == "regression":
            return nn.MSELoss()
        if self.task_type == "multiclass_classification":
            return nn.CrossEntropyLoss()
        return nn.BCEWithLogitsLoss()

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(self.device) if torch.is_tensor(value) else value
        return moved

    def _prepare_logits_and_labels(self, logits: torch.Tensor, labels: torch.Tensor):
        if self.task_type == "multiclass_classification":
            return logits, labels.long()
        return logits.reshape(-1), labels.float().reshape(-1)

    def _scores_from_logits(self, logits: torch.Tensor) -> np.ndarray:
        if self.task_type == "multiclass_classification":
            return torch.softmax(logits, dim=-1).detach().cpu().numpy()
        if self.task_type == "binary_classification":
            return torch.sigmoid(logits.reshape(-1)).detach().cpu().numpy()
        return logits.reshape(-1).detach().cpu().numpy()

    def train_epoch(self, train_loader):
        self.model.train()
        return self._run_epoch(train_loader, training=True)

    def evaluate_epoch(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            return self._run_epoch(test_loader, training=False)

    def _run_epoch(self, data_loader, training: bool) -> dict[str, Any]:
        total_loss = 0.0
        num_batches = 0
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        desc = "Training" if training else "Evaluating"

        for batch in tqdm(data_loader, desc=desc):
            batch = self._move_batch_to_device(batch)
            outputs = self.model(batch)
            if "logits" not in outputs:
                raise ValueError("Model forward must return a dict containing 'logits'.")

            logits, labels = self._prepare_logits_and_labels(outputs["logits"], batch["label"])
            loss = self.criterion(logits, labels)

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1
            all_scores.append(self._scores_from_logits(logits))
            all_labels.append(labels.detach().cpu().numpy())

        if num_batches == 0:
            raise ValueError("DataLoader produced no batches.")

        scores = np.concatenate(all_scores, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        avg_loss = total_loss / num_batches
        return {
            "loss": avg_loss,
            "metrics": self.calculate_metrics(scores, labels),
        }

    def calculate_metrics(self, scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        return self.evaluator.evaluate(scores, labels)

    def _monitor_value(self, eval_results: dict[str, Any]) -> float:
        metric_name = self.monitor_metric
        if metric_name in {"loss", "val_loss"}:
            return float(eval_results["loss"])
        key = metric_name[4:] if metric_name.startswith("val_") else metric_name
        if key not in eval_results["metrics"]:
            self.log_and_print(f"[WARN] monitor_metric={metric_name} not found; using val_loss.")
            return float(eval_results["loss"])
        return float(eval_results["metrics"][key])

    def _is_better(self, current: float, best: float) -> bool:
        return current > best if self.monitor_mode == "max" else current < best

    def save_history(self, epoch: int, train_results: dict[str, Any], eval_results: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "epoch": epoch + 1,
            "train_loss": train_results["loss"],
            "val_loss": eval_results["loss"],
            **{f"train_{key}": value for key, value in train_results["metrics"].items()},
            **{f"val_{key}": value for key, value in eval_results["metrics"].items()},
            "lr": self.optimizer.param_groups[0]["lr"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.training_history.append(entry)
        pd.DataFrame(self.training_history).to_csv(self.history_file, index=False)
        return entry

    def print_epoch_results(
        self,
        epoch: int,
        train_results: dict[str, Any],
        eval_results: dict[str, Any],
        monitor_value: float,
        epoch_time: float,
    ) -> None:
        val_metrics = ", ".join(f"{key}={value:.4f}" for key, value in eval_results["metrics"].items())
        train_metrics = ", ".join(f"{key}={value:.4f}" for key, value in train_results["metrics"].items())
        self.log_and_print(
            f"Epoch {epoch + 1}/{self.epochs} | "
            f"train_loss={train_results['loss']:.4f} | val_loss={eval_results['loss']:.4f} | "
            f"monitor={monitor_value:.4f} | lr={self.optimizer.param_groups[0]['lr']:.6f} | "
            f"time={epoch_time:.2f}s\n"
            f"train_metrics: {train_metrics}\n"
            f"val_metrics: {val_metrics}"
        )

    def train_from_epoch(self, train_loader, test_loader, start_epoch=0):
        best_value = -float("inf") if self.monitor_mode == "max" else float("inf")
        best_entry: dict[str, Any] | None = None

        self.log_and_print(f"Train batches: {len(train_loader)}")
        self.log_and_print(f"Eval batches: {len(test_loader)}")

        for epoch in range(start_epoch, self.epochs):
            start_time = time.time()
            train_results = self.train_epoch(train_loader)
            eval_results = self.evaluate_epoch(test_loader)
            monitor_value = self._monitor_value(eval_results)
            self.scheduler.step(monitor_value)

            entry = self.save_history(epoch, train_results, eval_results)
            epoch_time = time.time() - start_time
            self.print_epoch_results(epoch, train_results, eval_results, monitor_value, epoch_time)
            self.log_event(
                "epoch_end",
                {
                    "epoch": epoch + 1,
                    "monitor_metric": self.monitor_metric,
                    "monitor_value": monitor_value,
                    "epoch_time_seconds": epoch_time,
                    "history_entry": entry,
                },
            )

            if self._is_better(monitor_value, best_value):
                best_value = monitor_value
                best_entry = entry
                self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "monitor_metric": self.monitor_metric,
                        "monitor_value": monitor_value,
                        "data_meta": self.data_meta,
                        "training_history": self.training_history,
                    },
                    self.best_model_path,
                )
                self.log_and_print(f"Saved best checkpoint: {self.best_model_path}")
                self.log_event(
                    "checkpoint_saved",
                    {
                        "epoch": epoch + 1,
                        "monitor_metric": self.monitor_metric,
                        "monitor_value": monitor_value,
                        "checkpoint_file": self.best_model_path,
                    },
                )

            if self.early_stopper.step(monitor_value):
                self.log_and_print(f"Early stopping at epoch {epoch + 1}.")
                self.log_event("early_stopping", {"epoch": epoch + 1, "monitor_value": monitor_value})
                break

        report_path = self._write_report(best_entry)
        predictions_path = self._write_predictions(test_loader)
        metrics_path = self._write_metrics_json(best_entry, report_path, predictions_path)
        manifest_path = self._write_artifact_manifest(best_entry, report_path, predictions_path, metrics_path)
        write_run_metadata(
            self.run_context,
            self.cfg,
            self.data_meta,
            status="completed",
            extra={
                "history_file": str(self.history_file),
                "config_snapshot_file": str(self.config_snapshot_file),
                "checkpoint_file": str(self.best_model_path),
                "report_file": str(report_path) if report_path else "",
                "events_file": str(self.events_file),
                "metrics_file": str(metrics_path) if metrics_path else "",
                "predictions_file": str(predictions_path) if predictions_path else "",
                "artifact_manifest_file": str(manifest_path) if manifest_path else "",
                "best_entry": best_entry or {},
            },
        )
        self.log_event(
            "run_completed",
            {
                "history_file": self.history_file,
                "checkpoint_file": self.best_model_path,
                "report_file": report_path,
                "metrics_file": metrics_path,
                "predictions_file": predictions_path,
                "artifact_manifest_file": manifest_path,
            },
        )
        self.log_and_print("Training complete.")
        self.log_and_print(f"history: {self.history_file}")
        self.log_and_print(f"checkpoint: {self.best_model_path}")
        if report_path:
            self.log_and_print(f"report: {report_path}")
        if manifest_path:
            self.log_and_print(f"artifacts_manifest: {manifest_path}")
        return best_entry

    def _write_report(self, best_entry: dict[str, Any] | None) -> Path | None:
        report_cfg = self.cfg.get("report", {})
        if not bool(report_cfg.get("enabled", True)):
            return None
        output_dir = Path(str(report_cfg.get("output_dir", "result/reports")))
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_name = str(self.cfg.get("data", {}).get("name", "dataset"))
        model_name = str(getattr(self.model, "name", self.model.__class__.__name__)).lower()
        report_path = output_dir / f"{data_name}_{model_name}_{timestamp}.md"
        generate_markdown_report(
            cfg=self.cfg,
            data_meta=self.data_meta,
            history_file=self.history_file,
            config_snapshot_file=self.config_snapshot_file,
            checkpoint_file=self.best_model_path,
            report_path=report_path,
            best_entry=best_entry,
            run_context=self.run_context,
        )
        return report_path

    def _write_predictions(self, data_loader) -> Path | None:
        if not self.save_predictions:
            return None

        self.model.eval()
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_sample_ids: list[str] = []
        auto_index = 0

        with torch.no_grad():
            for batch in data_loader:
                sample_ids = batch.get("sample_id")
                batch = self._move_batch_to_device(batch)
                outputs = self.model(batch)
                if "logits" not in outputs:
                    raise ValueError("Model forward must return a dict containing 'logits'.")
                logits, labels = self._prepare_logits_and_labels(outputs["logits"], batch["label"])
                scores = self._scores_from_logits(logits)
                labels_np = labels.detach().cpu().numpy()
                all_scores.append(scores)
                all_labels.append(labels_np)

                batch_size = labels_np.shape[0]
                if sample_ids is None:
                    all_sample_ids.extend(str(idx) for idx in range(auto_index, auto_index + batch_size))
                    auto_index += batch_size
                elif isinstance(sample_ids, (list, tuple)):
                    all_sample_ids.extend(str(item) for item in sample_ids)
                else:
                    all_sample_ids.extend(str(item) for item in list(sample_ids))

        scores = np.concatenate(all_scores, axis=0)
        labels = np.concatenate(all_labels, axis=0).reshape(-1)
        rows: dict[str, Any] = {
            "sample_id": all_sample_ids,
            "label": labels,
        }

        if self.task_type == "multiclass_classification":
            predictions = np.argmax(scores, axis=1)
            rows["prediction"] = predictions
            for class_idx in range(scores.shape[1]):
                rows[f"score_class_{class_idx}"] = scores[:, class_idx]
        elif self.task_type == "binary_classification":
            threshold = get_threshold(self.cfg.get("metrics", {}), self.task_type)
            scores = scores.reshape(-1)
            rows["prediction"] = (scores >= threshold).astype(int)
            rows["score"] = scores
        else:
            scores = scores.reshape(-1)
            rows["prediction"] = scores

        pd.DataFrame(rows).to_csv(self.predictions_file, index=False)
        return self.predictions_file

    def _write_metrics_json(
        self,
        best_entry: dict[str, Any] | None,
        report_path: Path | None,
        predictions_path: Path | None,
    ) -> Path | None:
        if not self.save_metrics_json:
            return None
        final_entry = self.training_history[-1] if self.training_history else {}
        payload = {
            "experiment_id": self.run_context.get("experiment_id", "n/a"),
            "run_id": self.run_context.get("run_id", "n/a"),
            "task_type": self.task_type,
            "monitor_metric": self.monitor_metric,
            "monitor_mode": self.monitor_mode,
            "best_entry": best_entry or {},
            "final_entry": final_entry,
            "artifacts": {
                "history_file": self.history_file,
                "config_snapshot_file": self.config_snapshot_file,
                "checkpoint_file": self.best_model_path,
                "report_file": report_path,
                "predictions_file": predictions_path,
                "events_file": self.events_file,
            },
        }
        self.metrics_file.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return self.metrics_file

    def _write_artifact_manifest(
        self,
        best_entry: dict[str, Any] | None,
        report_path: Path | None,
        predictions_path: Path | None,
        metrics_path: Path | None,
    ) -> Path | None:
        if not self.save_manifest:
            return None
        payload = {
            "experiment_id": self.run_context.get("experiment_id", "n/a"),
            "run_id": self.run_context.get("run_id", "n/a"),
            "status": "completed",
            "task_type": self.task_type,
            "data": {
                "name": self.cfg.get("data", {}).get("name", "unknown"),
                "path": self.cfg.get("data", {}).get("path", "unknown"),
                "label_col": self.data_meta.get("label_col", "label"),
                "train_size": self.data_meta.get("train_size"),
                "test_size": self.data_meta.get("test_size"),
            },
            "model": getattr(self.model, "name", self.model.__class__.__name__),
            "monitor_metric": self.monitor_metric,
            "monitor_mode": self.monitor_mode,
            "best_entry": best_entry or {},
            "artifacts": {
                "text_log": self.log_file,
                "events_jsonl": self.events_file,
                "history_csv": self.history_file,
                "metrics_json": metrics_path,
                "predictions_csv": predictions_path,
                "resolved_config_yaml": self.config_snapshot_file,
                "checkpoint": self.best_model_path,
                "report_md": report_path,
                "tracking_metadata_json": Path(str(self.run_context.get("run_dir", ""))) / "run_metadata.json"
                if self.run_context.get("run_dir")
                else None,
            },
        }
        self.manifest_file.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return self.manifest_file

    def train(self, train_loader, test_loader):
        return self.train_from_epoch(train_loader, test_loader, start_epoch=0)


class BinaryClassificationTrainer(BaseTrainer):
    expected_task_type = "binary_classification"


class MulticlassClassificationTrainer(BaseTrainer):
    expected_task_type = "multiclass_classification"


class RegressionTrainer(BaseTrainer):
    expected_task_type = "regression"
