# Verification

Use this reference before claiming a paper method is ready.

## Lightweight Checks First

```bash
python scripts/doctor.py data=<dataset> model=<method_name> trainer=<trainer> metrics=default
python -m py_compile main.py Data_pre.py Dataset.py engine.py
```

If the paper method adds new files, compile them too:

```bash
python -m py_compile models/<method_name>.py
```

## Tiny Smoke Run

Run only when local resources allow it:

```bash
python main.py data=<dataset> model=<method_name> trainer=<trainer> metrics=default training_loop.epochs=1
```

On a weak local machine, skip full training and state exactly which checks were run.

## If A Core Contract Changed

Run the matching sample doctor checks:

```bash
python scripts/doctor.py data=sample_binary model=transformer trainer=binary metrics=default training_loop.epochs=1
python scripts/doctor.py data=sample_multiclass model=transformer trainer=multiclass metrics=default training_loop.epochs=1
python scripts/doctor.py data=sample_regression model=transformer trainer=regression metrics=default training_loop.epochs=1
```

## Tuning, Ablation, And Comparison

Validate orchestration with dry-runs before launching real jobs:

```bash
python scripts/tuning/run_grid_search.py configs/tuning/<method_name>_grid.yaml --dry-run --max-runs 2
python scripts/ablation/run_ablation.py configs/ablation/<method_name>_ablation.yaml --dry-run --max-runs 2
python scripts/experiment/run_experiment.py configs/experiment/<comparison>.yaml --dry-run --max-runs 2
```

## Final Handoff

Report:

- files changed
- paper method folder
- model/trainer/data adapter choices
- validation commands run
- commands intentionally not run
