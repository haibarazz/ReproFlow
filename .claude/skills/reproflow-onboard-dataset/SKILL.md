---
name: reproflow-onboard-dataset
description: Add a custom dataset to ReproFlow using the dataset contract and validate it before training.
---

# ReproFlow Dataset Onboarding

Use this skill when a user wants to bring their own dataset into ReproFlow.

## Steps

1. Put the data file under `dataset/<task_name>/` or `dataset/<task_name>.csv`.
2. Create `configs/data/<task_name>.yaml`.
3. Declare:
   - `name`
   - `path`
   - `task_type`
   - `id_col` if available
   - `label_col`
   - `numeric_cols`
   - `categorical_cols`
   - `text_cols`
   - `split`
   - `preprocess`
4. Do not edit `Data_pre.py` for normal tabular or text-feature datasets.
5. Validate the contract:

```bash
python scripts/doctor.py data=<task_name> model=transformer trainer=<binary|multiclass|regression> metrics=default
```

## Common Fixes

- If a column is missing, fix `configs/data/<task_name>.yaml`, not model code.
- If labels are strings, let the preprocessing pipeline encode them.
- If binary classification has more than two labels, change the task to `multiclass_classification` or fix the label definition.
- Use `seed_controls_data_split: false` for model-seed comparisons unless the experiment explicitly studies split variance.
