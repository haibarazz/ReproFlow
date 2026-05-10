from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml


def _snake_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_").lower()
    return cleaned or "paper_method"


def _class_name(value: str) -> str:
    return "".join(part.capitalize() for part in _snake_name(value).split("_")) + "Model"


def _write(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        print(f"skip existing: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"write: {path}")


def _method_yaml(
    method_name: str,
    paper: str | None,
    dataset: str,
    task_type: str,
    trainer: str,
    model_config: str,
) -> str:
    payload: dict[str, Any] = {
        "method_name": method_name,
        "implementation_status": "scaffolded",
        "paper": {
            "title": "TODO",
            "venue": "TODO",
            "year": "TODO",
            "local_path": paper or "docs/papers/TODO.pdf",
        },
        "task": {
            "type": task_type,
            "target_dataset": dataset,
            "trainer": trainer,
        },
        "method_contract": {
            "model_config": model_config,
            "model_forward": 'forward(batch) -> {"logits": logits}',
            "required_batch_keys": ["basic_features", "label"],
            "new_dataset_keys": [],
        },
        "baselines": ["traditional_ml", "standard_deep_model"],
        "validation_commands": [
            f"python main.py data={dataset} model={method_name} trainer={trainer} training_loop.epochs=1",
            "python -m py_compile main.py Data_pre.py Dataset.py engine.py reproflow/reports.py",
        ],
        "reporting": {
            "single_run_report": "result/reports/",
            "aggregate_report_command": "python scripts/reports/generate_experiment_report.py",
        },
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def scaffold(
    method_name: str,
    *,
    paper: str | None,
    dataset: str,
    task_type: str,
    trainer: str,
    force: bool,
) -> None:
    module_name = _snake_name(method_name)
    class_name = _class_name(method_name)
    method_dir = Path("paper_methods") / module_name
    model_config = f"configs/model/{module_name}.yaml"

    _write(
        method_dir / "method.yaml",
        _method_yaml(module_name, paper, dataset, task_type, trainer, model_config),
        force,
    )
    _write(
        method_dir / "implementation_checklist.md",
        f"""# {module_name} Implementation Checklist

- [ ] Read the target paper in `{paper or "docs/papers/TODO.pdf"}`.
- [ ] Map the paper task to `{task_type}` on dataset `{dataset}`.
- [ ] Implement `{class_name}` in `models/{module_name}.py`.
- [ ] Keep model forward output as `{{"logits": logits}}`.
- [ ] Update `{model_config}` if constructor arguments change.
- [ ] Run one smoke experiment:

```bash
python main.py data={dataset} model={module_name} trainer={trainer} training_loop.epochs=1
```

- [ ] Run or update the tuning spec under `configs/tuning/{module_name}_grid.yaml`.
- [ ] If the method needs ablation, update `configs/ablation/{module_name}_ablation.yaml`.
- [ ] Generate an aggregate report:

```bash
python scripts/reports/generate_experiment_report.py
```
""",
        force,
    )
    _write(
        method_dir / "README.md",
        f"""# {module_name}

This folder tracks the paper-to-implementation mapping for `{module_name}`.

## Paper

- Local file: `{paper or "docs/papers/TODO.pdf"}`
- Core idea: TODO
- Difference from original paper setting: TODO

## ReproFlow Integration

- Dataset config: `configs/data/{dataset}.yaml`
- Model config: `{model_config}`
- Trainer: `{trainer}`
- Task type: `{task_type}`

## Command

```bash
python main.py data={dataset} model={module_name} trainer={trainer}
```
""",
        force,
    )
    _write(
        method_dir / "run_report_template.md",
        f"""# {module_name} Reproduction Report

## Paper

TODO

## Dataset

- Dataset config: `configs/data/{dataset}.yaml`
- Task type: `{task_type}`

## Implementation Mapping

TODO

## Main Results

TODO: paste single-run, baseline, multi-seed, and ablation summaries.

## Reproduction Gap

TODO: explain differences between the original paper setting and this dataset.
""",
        force,
    )
    _write(
        Path(model_config),
        f"""_target_: models.{module_name}.{class_name}
hidden_dim: 128
dropout: 0.1
""",
        force,
    )
    _write(
        Path("configs/tuning") / f"{module_name}_grid.yaml",
        f"""main_script: main.py
config_name: config
tuning_name: {module_name}_grid

choices:
  data: {dataset}
  model: {module_name}
  trainer: {trainer}
  training_loop: default
  metrics: default

monitor_metric: val_loss
monitor_mode: min
output_root: result/tuning
seeds: [42, 43, 44]
seed_overrides:
  - random.seed
seed_controls_data_split: false

base_overrides:
  training_loop.epochs: 2
  training_loop.batch_size: 128

grid:
  model.hidden_dim: [64, 128]
  model.dropout: [0.1, 0.3]
  training_loop.learning_rate: [0.001, 0.0005]
""",
        force,
    )
    _write(
        Path("configs/ablation") / f"{module_name}_ablation.yaml",
        f"""main_script: main.py
config_name: config
ablation_name: {module_name}_ablation

choices:
  data: {dataset}
  model: {module_name}
  trainer: {trainer}
  training_loop: default
  metrics: default

monitor_metric: val_loss
monitor_mode: min
output_root: result/ablation
seeds: [42, 43, 44]
seed_overrides:
  - random.seed
seed_controls_data_split: false

base_overrides:
  training_loop.epochs: 2
  training_loop.batch_size: 128

variants:
  - name: full
    overrides: {{}}
  - name: small_hidden
    overrides:
      model.hidden_dim: 64
""",
        force,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold a ReproFlow paper method.")
    parser.add_argument("method_name", help="Paper method name, for example paper_xxx")
    parser.add_argument("--paper", default=None, help="Local paper path under docs/papers/")
    parser.add_argument("--dataset", default="sample_binary", help="Target data config name")
    parser.add_argument("--task-type", default="binary_classification")
    parser.add_argument("--trainer", default="binary")
    parser.add_argument("--force", action="store_true", help="Overwrite existing scaffold files")
    args = parser.parse_args()

    scaffold(
        args.method_name,
        paper=args.paper,
        dataset=args.dataset,
        task_type=args.task_type,
        trainer=args.trainer,
        force=args.force,
    )


if __name__ == "__main__":
    main()
