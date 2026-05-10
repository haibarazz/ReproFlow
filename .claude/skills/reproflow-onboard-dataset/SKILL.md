---
name: reproflow-onboard-dataset
description: Add a custom dataset to ReproFlow by writing a data YAML, choosing the correct adapter/Dataset contract, and validating it before training.
---

# ReproFlow Dataset Onboarding

Use this skill when a user wants to bring their own dataset into ReproFlow.

## Progressive Disclosure

- Normal CSV/tabular dataset: follow the quick path below.
- New batch shape or paper-specific preprocessing: read `references/adapter-patterns.md`.
- Paper reproduction context: also read `../reproflow-reproduce-paper/references/contracts.md`.

## Quick Path

1. Put the data file under `dataset/<task_name>/` or `dataset/<task_name>.csv`.
2. Create `configs/data/<task_name>.yaml`.
3. Declare data path, label, feature columns, split, preprocess, `adapter`, and `dataset`.
4. For normal tabular or text-feature datasets, use:

```yaml
adapter:
  _target_: reproflow.data.tabular.TabularDataAdapter
dataset:
  _target_: reproflow.data.tabular.TabularDataset
```

5. For recommender, graph, sequence, or paper-specific batch shapes, add a focused adapter/Dataset under `reproflow/data/` or `paper_methods/<method>/data.py`.
6. Do not edit `Data_pre.py` or `Dataset.py` for normal dataset onboarding.
7. Validate the contract:

```bash
python scripts/doctor.py data=<task_name> model=transformer trainer=<binary|multiclass|regression> metrics=default
```

## Common Fixes

- If a column is missing, fix `configs/data/<task_name>.yaml`, not model code.
- If labels are strings, let the preprocessing pipeline encode them.
- If binary classification has more than two labels, change the task to `multiclass_classification` or fix the label definition.
- Use `seed_controls_data_split: false` for model-seed comparisons unless the experiment explicitly studies split variance.
- Use `configs/data/examples/recommender_pairwise_example.yaml` and `configs/data/examples/graph_minibatch_example.yaml` as examples when a method needs a different batch contract.
