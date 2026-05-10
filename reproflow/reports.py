from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf


def _format_metrics(entry: dict[str, Any] | None) -> str:
    if not entry:
        return "No completed epoch was recorded."
    metric_items = []
    for key, value in entry.items():
        if key.startswith("val_") and key != "val_loss":
            try:
                metric_items.append(f"- `{key}`: {float(value):.6f}")
            except (TypeError, ValueError):
                metric_items.append(f"- `{key}`: {value}")
    if not metric_items:
        metric_items.append(f"- `val_loss`: {float(entry.get('val_loss', 0.0)):.6f}")
    return "\n".join(metric_items)


def generate_markdown_report(
    cfg,
    data_meta: dict[str, Any],
    history_file: Path,
    config_snapshot_file: Path,
    checkpoint_file: Path,
    report_path: Path,
    best_entry: dict[str, Any] | None,
    run_context: dict[str, Any] | None = None,
) -> None:
    history = pd.read_csv(history_file) if Path(history_file).exists() else pd.DataFrame()
    last_entry = history.iloc[-1].to_dict() if not history.empty else {}
    data_cfg = cfg.get("data", {})
    model_choice = cfg.get("model", {}).get("_target_", "unknown")
    trainer_choice = cfg.get("trainer", {}).get("_target_", "unknown")
    run_context = run_context or {}

    content = f"""# ReproFlow Experiment Report

## Run Summary

- Dataset: `{data_cfg.get("name", "unknown")}`
- Data path: `{data_cfg.get("path", "unknown")}`
- Task type: `{data_meta.get("task_type", "unknown")}`
- Model config: `{model_choice}`
- Trainer config: `{trainer_choice}`
- Experiment ID: `{run_context.get("experiment_id", "n/a")}`
- Run ID: `{run_context.get("run_id", "n/a")}`
- Seed: `{run_context.get("seed", cfg.get("random", {}).get("seed", "n/a"))}`
- Train/Test size: `{data_meta.get("train_size", "?")}` / `{data_meta.get("test_size", "?")}`
- Feature count: `{len(data_meta.get("feature_cols", []))}`

## Best Validation Metrics

{_format_metrics(best_entry)}

## Final Epoch

- `epoch`: {last_entry.get("epoch", "n/a")}
- `train_loss`: {last_entry.get("train_loss", "n/a")}
- `val_loss`: {last_entry.get("val_loss", "n/a")}

## Artifacts

- History CSV: `{history_file}`
- Resolved config: `{config_snapshot_file}`
- Best checkpoint: `{checkpoint_file}`
- Tracking metadata: `{Path(str(run_context.get("run_dir", ""))) / "run_metadata.json" if run_context.get("run_dir") else "n/a"}`
- Structured run directory: `{run_context.get("run_dir", "n/a")}`

## Dataset Contract

- Label column: `{data_meta.get("label_col", "label")}`
- Numeric columns: `{data_meta.get("numeric_cols", [])}`
- Categorical columns: `{data_meta.get("categorical_cols", [])}`
- Text columns: `{data_meta.get("text_cols", [])}`

## Resolved Config

```yaml
{OmegaConf.to_yaml(cfg, resolve=True)}
```
"""
    report_path.write_text(content, encoding="utf-8")
