---
name: reproflow-debug-run
description: Diagnose failed ReproFlow runs using configs, doctor checks, logs, and generated artifacts.
---

# ReproFlow Debug Run

Use this skill when a training, tuning, ablation, experiment, or benchmark run fails.

## First Checks

```bash
python scripts/doctor.py data=<dataset> model=<model> trainer=<trainer> metrics=default
```

Then inspect:

- `result/**/training_*.log`
- `result/**/config_*.yaml`
- `result/**/history_*.csv`
- `result/**/summary_latest.csv`

Optional structured artifacts exist only when explicitly enabled:

- `result/tracking/<experiment_id>/<run_id>/run_metadata.json`
- `result/tracking/<experiment_id>/<run_id>/artifacts_manifest.json`
- `result/tracking/<experiment_id>/<run_id>/events.jsonl`
- `result/tracking/<experiment_id>/<run_id>/metrics_latest.json`
- `result/tracking/<experiment_id>/<run_id>/predictions.csv`

## Common Failure Modes

- Dataset config references a missing column.
- `task_type` does not match trainer.
- `training_loop.monitor_metric` is not included in the selected metrics.
- Model forward does not return `{"logits": logits}`.
- Multiclass logits shape is not `(batch, num_classes)`.
- Binary/regression logits have an extra incompatible dimension.

## Fix Order

1. Fix config mistakes first.
2. Fix model forward contract second.
3. Change trainer only if the method objective truly requires it.
4. Re-run doctor before running a full experiment.
