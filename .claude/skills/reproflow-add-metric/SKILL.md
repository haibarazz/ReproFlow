---
name: reproflow-add-metric
description: Add or configure metrics without modifying the training loop.
---

# ReproFlow Add Metric

Use this skill when adding a new evaluation metric or metric set.

## Where Metrics Live

- Binary and multiclass metrics: `metrics/classification.py`
- Regression metrics: `metrics/regression.py`
- Ranking placeholders: `metrics/ranking.py`
- Registry and defaults: `metrics/registry.py`
- User-facing config: `configs/metrics/*.yaml`

## Rules

- Do not add metric code to `engine.py`.
- Keep metric functions pure: inputs are scores, labels, and metric names.
- Return a flat `dict[str, float]`.
- Add the metric name to `SUPPORTED_METRICS` through the task-specific metric set.
- Add or update a `configs/metrics/*.yaml` file if users should select the metric.

## Verification

```bash
python scripts/doctor.py data=sample_binary model=transformer trainer=binary metrics=default
python main.py data=sample_binary model=transformer trainer=binary metrics=default training_loop.epochs=1
```

For task-specific metrics, also run that task's sample command.
