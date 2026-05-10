---
name: reproflow-run-fair-experiment
description: Run or create fair baseline-vs-paper-method comparisons with shared metrics and seeds.
---

# ReproFlow Fair Experiment

Use this skill when comparing baselines, paper methods, tuning results, or ablations.

## Manifest

Create or edit `configs/experiment/<name>.yaml`:

```yaml
experiment_name: <name>
data: <dataset>
trainer: <trainer>
training_loop: default
metrics: default
seeds: [42, 43, 44]
monitor_metric: val_auc
monitor_mode: max
benchmark_metric: roc_auc
methods:
  - name: traditional_ml
    type: ml_benchmark
    primary_metric: roc_auc
  - name: paper_method
    type: deep_learning
    model: paper_method
```

## Run

Dry-run first:

```bash
python scripts/experiment/run_experiment.py configs/experiment/<name>.yaml --dry-run --max-runs 2
```

Then run:

```bash
python scripts/experiment/run_experiment.py configs/experiment/<name>.yaml
```

Generate aggregate report:

```bash
python scripts/reports/generate_experiment_report.py
```

## Fairness Rules

- Use the same dataset config.
- Use the same metric config.
- Use the same seed list.
- Keep `seed_controls_data_split: false` unless intentionally testing split variance.
- Report mean and std across seeds.
