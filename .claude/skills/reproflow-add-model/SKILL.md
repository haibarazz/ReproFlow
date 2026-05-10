---
name: reproflow-add-model
description: Add a new deep learning model or baseline to ReproFlow while preserving model, config, data, and trainer contracts.
---

# ReproFlow Add Model

Use this skill when adding a baseline, paper method, or architecture variant.

## Before Coding

- For paper reproduction, read `../reproflow-reproduce-paper/references/paper-workflow.md`.
- For model/data/trainer contracts, read `../reproflow-reproduce-paper/references/contracts.md`.
- If the model needs a non-standard batch or loss, read `../reproflow-reproduce-paper/references/decision-guide.md`.

## Required Files

- `models/<model_name>.py` or `paper_methods/<method_name>/model.py`
- `configs/model/<model_name>.yaml`
- optional `configs/tuning/<model_name>_grid.yaml`
- optional `configs/ablation/<model_name>_ablation.yaml`

## Model Constructor

The model should accept these common arguments when possible:

```python
def __init__(self, input_dim: int, task_type: str, num_classes: int = 1, ...):
    ...
```

## Forward Contract

```python
def forward(self, batch):
    features = batch["basic_features"]
    return {"logits": logits}
```

Expected output shape:

- binary classification: `(batch,)` or `(batch, 1)`
- multiclass classification: `(batch, num_classes)`
- regression: `(batch,)` or `(batch, 1)`

## Verification

```bash
python scripts/doctor.py data=sample_binary model=<model_name> trainer=binary metrics=default
python main.py data=sample_binary model=<model_name> trainer=binary metrics=default training_loop.epochs=1
```

If the model is task-specific, run the matching sample task only.
