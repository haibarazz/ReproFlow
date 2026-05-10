from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class MLBenchmarkRunner:
    """Batch runner for traditional ML baselines."""

    def __init__(
        self,
        task_type: str = "classification",
        models_to_run: list[str] | None = None,
        metrics_to_use: list[str] | None = None,
        save_dir: str = "ml_results",
        verbose: bool = False,
        save_format: dict[str, bool] | None = None,
        primary_metric: str | None = None,
        config_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.task_type = task_type
        self.models_to_run = models_to_run or []
        self.metrics_to_use = metrics_to_use or []
        self.save_dir = Path(save_dir)
        self.verbose = verbose
        self.save_format = save_format or {"csv": True, "json": True}
        self.primary_metric = primary_metric
        self.config_snapshot = config_snapshot or {}
        self.results: list[dict[str, Any]] = []
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def run_benchmark(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        from .evaluation.metrics import MetricsCalculator
        from .models import MODEL_REGISTRY

        print("=" * 60)
        print("Starting ML Benchmark")
        print(f"task_type: {self.task_type}")
        print(f"models: {len(self.models_to_run)}")
        print(f"metrics: {', '.join(self.metrics_to_use)}")
        print("=" * 60)

        for model_name in self.models_to_run:
            if model_name not in MODEL_REGISTRY:
                self._record_error(model_name, "model is not registered")
                continue

            model_wrapper = MODEL_REGISTRY[model_name]()
            if model_wrapper.task_type != self.task_type:
                print(f"Skipping {model_name}: task_type={model_wrapper.task_type} != {self.task_type}")
                continue

            print(f"\n--- {model_name} ---")
            try:
                start_time = time.time()
                model_wrapper.fit(X_train, y_train)
                train_time = time.time() - start_time

                start_time = time.time()
                y_pred = model_wrapper.predict(X_test)
                y_proba = model_wrapper.predict_proba(X_test)
                predict_time = time.time() - start_time

                metrics = MetricsCalculator.calculate(
                    y_test,
                    y_pred,
                    y_proba,
                    task_type=self.task_type,
                    metrics=self.metrics_to_use,
                    average="weighted",
                )

                result = {
                    "model_name": model_name,
                    "status": "ok",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "task_type": self.task_type,
                    "train_time_seconds": round(train_time, 4),
                    "predict_time_seconds": round(predict_time, 4),
                    **metrics,
                }
                if self.verbose:
                    result["model_params"] = json.dumps(model_wrapper.get_model_params(), default=str)
                self.results.append(result)
                self._print_model_result(result)
            except Exception as exc:  # keep benchmark running even if one baseline fails
                self._record_error(model_name, str(exc))

        saved_paths = self._save_results()
        self._print_summary()
        return self.results, saved_paths

    def _record_error(self, model_name: str, error: str) -> None:
        print(f"FAILED {model_name}: {error}")
        self.results.append(
            {
                "model_name": model_name,
                "status": "failed",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "task_type": self.task_type,
                "error": error,
            }
        )

    def _print_model_result(self, result: dict[str, Any]) -> None:
        metric_text = []
        for metric in self.metrics_to_use:
            value = result.get(metric)
            if value is None:
                continue
            metric_text.append(f"{metric}={float(value):.4f}")
        print(", ".join(metric_text))

    def _main_metric(self) -> str | None:
        if self.primary_metric:
            return self.primary_metric
        if self.task_type == "classification":
            for metric in ("roc_auc", "f1", "accuracy"):
                if metric in self.metrics_to_use:
                    return metric
        for metric in ("r2", "rmse", "mse", "mae"):
            if metric in self.metrics_to_use:
                return metric
        return self.metrics_to_use[0] if self.metrics_to_use else None

    @staticmethod
    def _higher_is_better(metric: str) -> bool:
        return metric not in {"mse", "rmse", "mae"}

    def _save_results(self) -> dict[str, Path]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame(self.results)
        saved_paths: dict[str, Path] = {}

        if self.save_format.get("csv", False):
            csv_path = self.save_dir / f"benchmark_{timestamp}.csv"
            latest_path = self.save_dir / "benchmark_latest.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            df.to_csv(latest_path, index=False, encoding="utf-8-sig")
            saved_paths["csv"] = csv_path
            saved_paths["latest_csv"] = latest_path
            print(f"Saved CSV: {csv_path}")

        if self.save_format.get("json", False):
            json_path = self.save_dir / f"benchmark_{timestamp}.json"
            latest_json_path = self.save_dir / "benchmark_latest.json"
            json_payload = {
                "config": self.config_snapshot,
                "results": self.results,
            }
            json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            latest_json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            saved_paths["json"] = json_path
            saved_paths["latest_json"] = latest_json_path
            print(f"Saved JSON: {json_path}")

        return saved_paths

    def _print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("Benchmark Summary")
        print("=" * 60)
        metric = self._main_metric()
        ok_results = [row for row in self.results if row.get("status") == "ok" and metric in row]
        if metric and ok_results:
            reverse = self._higher_is_better(metric)
            ranked = sorted(ok_results, key=lambda row: row[metric], reverse=reverse)
            print(f"Ranked by {metric}:")
            for idx, row in enumerate(ranked[:10], 1):
                print(f"{idx:2d}. {row['model_name']:30s} {float(row[metric]):.4f}")

        failed = [row for row in self.results if row.get("status") == "failed"]
        if failed:
            print(f"\nFailed models: {len(failed)}")
            for row in failed:
                print(f"- {row['model_name']}: {row.get('error')}")
        print("=" * 60)
