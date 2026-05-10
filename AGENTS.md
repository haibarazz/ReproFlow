# ReproFlow Agent Guide

This repository is an AI-native reproducible ML experiment framework. Agents should add new paper methods through configured contracts instead of writing standalone scripts.

## Core Architecture

Runtime spine:

- `main.py`: Hydra entrypoint for deep learning runs
- `Data_pre.py`: compatibility entrypoint that delegates to the configured data adapter
- `Dataset.py`: compatibility alias for the default tabular Dataset
- `reproflow/data/`: data adapters and Dataset contracts for tabular, recommender, graph, and future data shapes
- `engine.py`: trainer loop for binary, multiclass, and regression tasks
- `run_ml_benchmark.py`: traditional ML baseline entrypoint

Do not replace this project with Lightning, MLflow, W&B, or another framework unless the user explicitly asks for that migration.

## Required Workflow

Before adding or modifying a model:

1. Read `docs/architecture.md`.
2. If the task is paper reproduction, start from `.claude/skills/reproflow-reproduce-paper/SKILL.md`.
3. Read only the referenced skill files needed for the current step.
4. Run the relevant preflight check:

```bash
python scripts/doctor.py data=<dataset> model=<model> trainer=<trainer> metrics=default
```

## Extension Rules

- Dataset schema belongs in `configs/data/*.yaml`.
- Metrics belong in `configs/metrics/*.yaml` and `metrics/`.
- Model hyperparameters belong in `configs/model/*.yaml`.
- Standard task trainers live in `configs/trainer/*.yaml` and `engine.py`.
- New non-standard trainers should live in focused modules instead of expanding `engine.py`.
- New data shapes should use `data.adapter` and `data.dataset` instead of expanding `Data_pre.py` or `Dataset.py`.
- Fair comparison groups belong in `configs/experiment/*.yaml`.
- Tuning belongs in `configs/tuning/*.yaml`.
- Ablation belongs in `configs/ablation/*.yaml`.
- Paper-specific documentation belongs in `paper_methods/<method_name>/`.

Do not hard-code dataset columns, label names, metrics, seeds, or file paths inside model code.

## Project Skills

Detailed reusable workflows live in `.claude/skills/`. They are plain Markdown and should be usable by any AI agent even when the host tool does not auto-load Claude skills.

Recommended skill by task:

- Add a dataset: `.claude/skills/reproflow-onboard-dataset/SKILL.md`
- Reproduce a paper: `.claude/skills/reproflow-reproduce-paper/SKILL.md`
- Add a model: `.claude/skills/reproflow-add-model/SKILL.md`
- Add a metric: `.claude/skills/reproflow-add-metric/SKILL.md`
- Run fair comparisons: `.claude/skills/reproflow-run-fair-experiment/SKILL.md`
- Debug a run: `.claude/skills/reproflow-debug-run/SKILL.md`

## Verification

For normal changes, run:

```bash
python -m py_compile main.py Data_pre.py Dataset.py engine.py run_ml_benchmark.py
python scripts/doctor.py data=sample_binary model=transformer trainer=binary metrics=default training_loop.epochs=1
```

For changes touching training behavior, also run one epoch for binary, multiclass, and regression sample tasks.

After any default training run, inspect the printed `training_*.log` and `history_*.csv` paths first. Structured artifacts are optional and should be enabled only when the task explicitly needs them.
