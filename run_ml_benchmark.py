"""
Traditional ML benchmark runner.

Examples:
    python run_ml_benchmark.py
    python run_ml_benchmark.py data=sample_regression task_type=regression
"""

from __future__ import annotations

import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Data_pre import get_data_config, normalize_task_type
from ml_benchmark.runner import MLBenchmarkRunner
from ml_benchmark.utils.data_loader import load_ml_data


def _benchmark_task_type(task_type: str) -> str:
    normalized = normalize_task_type(task_type)
    return "regression" if normalized == "regression" else "classification"


@hydra.main(config_path="configs", config_name="ml_benchmark", version_base=None)
def main(cfg: DictConfig):
    print("=" * 60)
    print("ReproFlow ML Benchmark Config")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    data_cfg = get_data_config(cfg)
    task_type = _benchmark_task_type(str(data_cfg.task_type))
    feature_cols = list(data_cfg.get("numeric_cols", [])) + list(data_cfg.get("categorical_cols", []))
    if data_cfg.get("text_cols"):
        print("[WARN] ML benchmark currently ignores text_cols. Use main.py for TF-IDF text features.")
    label_col = str(data_cfg.label_col)
    split_cfg = data_cfg.get("split", {})

    X_train, y_train, X_test, y_test = load_ml_data(
        cfg,
        feature_cols=feature_cols,
        label_col=label_col,
        train_ratio=float(split_cfg.get("train_ratio", 0.8)),
        random_state=int(split_cfg.get("random_state", 42)),
        cat_features=list(data_cfg.get("categorical_cols", [])),
        num_features=list(data_cfg.get("numeric_cols", [])),
    )

    from ml_benchmark.models import MODEL_REGISTRY

    models_to_run = [
        model_name for model_name, enabled in cfg.models.items()
        if enabled and model_name in MODEL_REGISTRY
    ]
    if not models_to_run:
        print("No benchmark models are enabled.")
        return

    from ml_benchmark.evaluation.metrics import MetricsCalculator

    valid_metric_registry = (
        MetricsCalculator.REGRESSION_METRICS
        if task_type == "regression"
        else MetricsCalculator.CLASSIFICATION_METRICS
    )
    metrics_to_use = [
        metric for metric, enabled in cfg.metrics.items()
        if enabled and metric in valid_metric_registry
    ]
    if not metrics_to_use:
        metrics_to_use = ["r2", "rmse", "mae"] if task_type == "regression" else ["accuracy", "f1", "roc_auc"]

    runner = MLBenchmarkRunner(
        task_type=task_type,
        models_to_run=models_to_run,
        metrics_to_use=metrics_to_use,
        save_dir=cfg.output.save_dir,
        verbose=cfg.output.verbose,
        save_format=cfg.output.save_format,
        primary_metric=cfg.output.get("primary_metric"),
        config_snapshot=OmegaConf.to_container(cfg, resolve=True),
    )
    runner.run_benchmark(X_train, y_train, X_test, y_test)
    print(f"Benchmark complete. Results saved to: {cfg.output.save_dir}")


if __name__ == "__main__":
    main()
