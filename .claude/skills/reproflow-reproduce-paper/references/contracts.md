# ReproFlow Contracts

Use this reference when adding files or checking whether an implementation fits the framework.

## Paper Method Folder

```text
paper_methods/<method_name>/
├── method.yaml
├── implementation_checklist.md
├── README.md
└── reproduction_notes.md
```

`method.yaml` should include:

- paper title
- task type
- expected inputs
- target label
- model config
- trainer config
- required baselines
- validation commands

## Model Contract

Preferred constructor:

```python
def __init__(self, input_dim: int, task_type: str, num_classes: int = 1, ...):
    ...
```

Forward contract:

```python
def forward(self, batch):
    return {"logits": logits}
```

Output shapes:

- binary classification: `(batch,)` or `(batch, 1)`
- multiclass classification: `(batch, num_classes)`
- regression: `(batch,)` or `(batch, 1)`

## Data Contract

Default batch keys:

```python
{
    "basic_features": Tensor,
    "label": Tensor,
    "sample_id": optional string
}
```

For ordinary CSV/tabular/text-feature tasks, use:

```yaml
adapter:
  _target_: reproflow.data.tabular.TabularDataAdapter
dataset:
  _target_: reproflow.data.tabular.TabularDataset
```

For new batch shapes, create a focused adapter/Dataset under `reproflow/data/` or a paper-local `paper_methods/<method_name>/data.py`.

## Config Contract

Add user-facing knobs to YAML, not hard-coded Python:

```text
configs/data/<dataset>.yaml
configs/model/<method_name>.yaml
configs/trainer/<trainer>.yaml
configs/tuning/<method_name>_grid.yaml
configs/ablation/<method_name>_ablation.yaml
configs/experiment/<comparison>.yaml
```

## Trainer Contract

Use the standard trainer unless the paper requires one of these:

- auxiliary loss
- pairwise/listwise ranking objective
- contrastive objective
- sequence generation objective
- graph-specific batch handling
- multi-task loss aggregation

Custom trainers should live outside `engine.py` when possible, for example in `reproflow/example_trainers.py` or a focused module.
