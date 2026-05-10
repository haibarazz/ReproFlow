# ReproFlow Architecture

ReproFlow keeps the original project style: `main.py`, `Data_pre.py`, `Dataset.py`, `engine.py`, `models/`, and `configs/` remain the core architecture. The upgrade is that data, task, trainer, and paper-method extension points are now explicit contracts instead of hidden assumptions.

## Agent Layer

`AGENTS.md` is the short universal entrypoint for AI coding agents. Project-local skills live in `.claude/skills/` and document repeatable workflows for dataset onboarding, paper reproduction, model addition, metric addition, fair experiments, and debugging.

## Runtime Flow

```text
configs/data/*.yaml
        |
        v
Data_pre.py
  - read CSV
  - validate columns
  - split train/test
  - encode labels
  - scale numeric features
  - encode categorical features
  - optionally build TF-IDF text features
        |
        v
Dataset.py
  - returns batch dict
  - basic_features
  - label
  - optional sample_id
        |
        v
models/
  - forward(batch) -> {"logits": tensor}
        |
        v
engine.py
  - BinaryClassificationTrainer
  - MulticlassClassificationTrainer
  - RegressionTrainer
        |
        v
evaluators/ + metrics/
  - task-aware metric registry
  - binary / multiclass / regression metrics
        |
        v
reproflow/reports.py
  - history csv
  - config snapshot
  - best checkpoint
  - markdown report
        |
        v
reproflow/tracking.py
  - local experiment_id / run_id metadata
        |
        v
scripts/tuning/run_grid_search.py / scripts/ablation/run_ablation.py / scripts/experiment/run_experiment.py
  - grid, candidate, ablation, fair comparison, and multi-seed runs over main.py
        |
        v
scripts/reports/generate_experiment_report.py
  - aggregate benchmark / tuning / ablation / seed summaries
```

## Extension Points

### Dataset

Add a new dataset by creating `configs/data/<name>.yaml` and placing the CSV under `dataset/`. No Python file should be edited for a normal tabular/text dataset.

### Model

Add a model by implementing a class that accepts `input_dim`, `task_type`, and `num_classes` when needed. The forward method must return `{"logits": logits}`.

### Trainer

Use existing trainers for binary classification, multiclass classification, and regression. Add a trainer only when the paper method needs a new loss, auxiliary objective, ranking metric, sequence objective, or graph-specific batch behavior.

### Metrics And Evaluator

Metrics live in `metrics/` and are selected by `configs/metrics/*.yaml`. Trainers should not grow task-specific metric code. The trainer sends scores and labels to `evaluators/Evaluator`, which dispatches to the metric registry.

Supported V1 metric families:

- binary classification: accuracy, precision, recall, F1, MCC, ROC-AUC, PR-AUC, Brier score
- multiclass classification: accuracy, balanced accuracy, macro/weighted precision/recall/F1, OVR/OVO AUC
- regression: MSE, RMSE, MAE, MAPE, median AE, R2, explained variance
- ranking placeholders: HitRate@K and NDCG@K for future ranking trainers

### Experiment

Experiment manifests live in `configs/experiment/*.yaml`. They define a fair comparison group over the same data, metrics, seeds, and output convention. They can run both deep learning methods through `main.py` and traditional ML baselines through `run_ml_benchmark.py`.

### Tuning

Deep learning tuning specs live in `configs/tuning/*.yaml`:

- `choices`: Hydra config groups for `data`, `model`, `trainer`, and `training_loop`
- `base_overrides`: fixed overrides applied to every run
- `grid`: cartesian product of parameter values
- `candidates`: optional explicit named override sets when a grid is too large
- `seeds`: optional first-class random seeds expanded across every candidate
- `seed_overrides`: config keys that receive each seed, normally `random.seed`
- `monitor_metric` and `monitor_mode`: summary ranking target

### Ablation

Ablation specs live in `configs/ablation/*.yaml`. They use named `variants` instead of broad grids. Each variant can be evaluated across the same `seeds` list, and the runner writes both per-run results and mean/std seed aggregates.

### Output And Optional Tracking

Default training output is intentionally small:

- `training_*.log`
- `history_*.csv`
- best checkpoint

Structured tracking is optional. Enable it only when a task needs richer artifacts:

```bash
python main.py tracking.enabled=true report.enabled=true artifacts.save_manifest=true
```

Optional artifacts include report markdown, predictions CSV, events JSONL, metrics JSON, and an artifact manifest.

### Doctor

`scripts/doctor.py` is the beginner/AI-agent preflight check. It validates:

- data file and schema columns
- task type and standard trainer match
- configured metrics are supported
- model can be instantiated
- model forward returns `{"logits": logits}`
- paper method folders have the required metadata files

### Paper Method

Each paper method should start from `paper_methods/template/` so the AI agent has a fixed place to document the paper mapping, implementation files, configs, and validation commands.

Use `scripts/paper_methods/scaffold.py` to create the first draft of:

- `paper_methods/<method>/method.yaml`
- `paper_methods/<method>/implementation_checklist.md`
- `configs/model/<method>.yaml`
- `configs/tuning/<method>_grid.yaml`
- `configs/ablation/<method>_ablation.yaml`

## Design Boundaries

ReproFlow is the experiment substrate. It should not become a giant deep learning framework. Heavy systems such as MLflow, Lightning, W&B, distributed training, or automatic PDF agents can be added later only if they serve the reproduction workflow.
