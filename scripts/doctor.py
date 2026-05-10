from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import to_absolute_path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Data_pre import build_data, get_data_config, normalize_task_type
from main import create_model
from metrics import get_metric_names, validate_metrics


def _record(items: list[str], level: str, message: str) -> None:
    items.append(f"[{level}] {message}")


def _target_name(cfg: Any) -> str:
    target = cfg.get("_target_", "") if hasattr(cfg, "get") else ""
    return str(target).split(".")[-1]


def _check_data_schema(cfg: Any, errors: list[str], warnings: list[str]) -> None:
    data_cfg = get_data_config(cfg)
    data_path = Path(to_absolute_path(str(data_cfg.path)))
    if not data_path.exists():
        _record(errors, "ERROR", f"Data file does not exist: {data_path}")
        return

    frame = pd.read_csv(data_path, nrows=20)
    required = [str(data_cfg.label_col)]
    if data_cfg.get("id_col"):
        required.append(str(data_cfg.id_col))
    required.extend(str(col) for col in data_cfg.get("numeric_cols", []))
    required.extend(str(col) for col in data_cfg.get("categorical_cols", []))
    required.extend(str(col) for col in data_cfg.get("text_cols", []))
    missing = [col for col in required if col not in frame.columns]
    if missing:
        _record(errors, "ERROR", f"Data schema missing columns: {missing}")
    else:
        _record(warnings, "OK", f"Data schema columns found in {data_path}")

    task_type = normalize_task_type(str(data_cfg.task_type))
    if task_type == "binary_classification":
        label_values = pd.read_csv(data_path, usecols=[str(data_cfg.label_col)])[str(data_cfg.label_col)].dropna().unique()
        if len(label_values) != 2:
            _record(errors, "ERROR", f"binary_classification expects exactly 2 labels, got {len(label_values)}")


def _check_trainer_task(cfg: Any, errors: list[str], warnings: list[str]) -> None:
    task_type = normalize_task_type(str(get_data_config(cfg).task_type))
    trainer_name = _target_name(cfg.trainer)
    expected = {
        "binary_classification": "BinaryClassificationTrainer",
        "multiclass_classification": "MulticlassClassificationTrainer",
        "regression": "RegressionTrainer",
    }.get(task_type)
    if expected and trainer_name != expected:
        _record(
            warnings,
            "WARN",
            f"Trainer {trainer_name} is not the standard trainer for task_type={task_type} ({expected}).",
        )
    else:
        _record(warnings, "OK", f"Trainer matches task_type={task_type}")


def _check_metrics(cfg: Any, errors: list[str], warnings: list[str]) -> None:
    task_type = normalize_task_type(str(get_data_config(cfg).task_type))
    unknown = validate_metrics(cfg.get("metrics", {}), task_type)
    if unknown:
        _record(errors, "ERROR", f"Unsupported metrics for {task_type}: {unknown}")
        return

    metric_names = get_metric_names(cfg.get("metrics", {}), task_type)
    _record(warnings, "OK", f"Metrics for {task_type}: {metric_names}")

    monitor_metric = str(cfg.training_loop.get("monitor_metric", "val_loss"))
    metric_key = monitor_metric[4:] if monitor_metric.startswith("val_") else monitor_metric
    if metric_key != "loss" and metric_key not in metric_names:
        _record(
            warnings,
            "WARN",
            f"training_loop.monitor_metric={monitor_metric} is not in configured metrics {metric_names}.",
        )


def _check_model_forward(cfg: Any, errors: list[str], warnings: list[str]) -> None:
    try:
        train_loader, _, input_dim, data_meta = build_data(cfg)
        model = create_model(cfg, input_dim=input_dim, data_meta=data_meta)
        model.eval()
        batch = next(iter(train_loader))
        with torch.no_grad():
            output = model(batch)
        if not isinstance(output, dict) or "logits" not in output:
            _record(errors, "ERROR", "Model forward must return a dict containing 'logits'.")
            return
        _record(warnings, "OK", f"Model forward contract valid, logits shape={tuple(output['logits'].shape)}")
    except Exception as exc:
        _record(errors, "ERROR", f"Model/data forward smoke failed: {exc}")


def _check_paper_methods(warnings: list[str]) -> None:
    methods_root = Path("paper_methods")
    if not methods_root.exists():
        _record(warnings, "WARN", "paper_methods/ does not exist.")
        return
    for method_dir in sorted(path for path in methods_root.iterdir() if path.is_dir() and path.name != "template"):
        method_yaml = method_dir / "method.yaml"
        checklist = method_dir / "implementation_checklist.md"
        if not method_yaml.exists():
            _record(warnings, "WARN", f"{method_dir} is missing method.yaml")
        if not checklist.exists():
            _record(warnings, "WARN", f"{method_dir} is missing implementation_checklist.md")


def run_doctor(overrides: list[str]) -> int:
    config_dir = str((Path.cwd() / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=overrides)

    errors: list[str] = []
    notes: list[str] = []
    _record(notes, "OK", f"Resolved config with overrides: {overrides or ['<none>']}")
    _check_data_schema(cfg, errors, notes)
    _check_trainer_task(cfg, errors, notes)
    _check_metrics(cfg, errors, notes)
    _check_model_forward(cfg, errors, notes)
    _check_paper_methods(notes)

    print("ReproFlow Doctor")
    print("=" * 80)
    for item in notes:
        print(item)
    for item in errors:
        print(item)
    print("=" * 80)
    if errors:
        print(f"Doctor failed with {len(errors)} error(s).")
        return 1
    print("Doctor passed.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ReproFlow data/model/trainer/metrics contracts.")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. data=sample_binary model=transformer trainer=binary")
    args = parser.parse_args()
    raise SystemExit(run_doctor(args.overrides))


if __name__ == "__main__":
    main()
