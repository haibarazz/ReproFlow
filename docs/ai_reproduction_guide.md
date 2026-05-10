# AI Reproduction Guide

This guide is the contract an AI coding agent must follow when reproducing a paper inside ReproFlow.

## Recommended Skill Entry

If the agent can read project-local skills, start with:

```text
.claude/skills/reproflow-reproduce-paper/SKILL.md
```

That skill gives the shortest workflow first and points to focused `references/` files only when the agent needs contracts, adapter/trainer decisions, or verification commands. Use this guide as the deeper project contract after the skill has identified the relevant path.

## Required Inputs

1. Target paper file under `docs/papers/`.
2. Dataset CSV under `dataset/`.
3. Dataset schema under `configs/data/<dataset_name>.yaml`.
4. A clear task type: `binary_classification`, `multiclass_classification`, or `regression`.

## Implementation Rules

- Do not write standalone training scripts for a paper method.
- Do not hard-code feature columns, label names, or file paths in Python.
- Use `configs/data/*.yaml` for data schema.
- Use `data.adapter` and `data.dataset` when the method needs a custom data shape.
- Use `configs/model/*.yaml` for model hyperparameters and `_target_`.
- Use `configs/metrics/*.yaml` for metrics. Do not add metric code to `engine.py`.
- Use `configs/trainer/*.yaml` only when the existing trainer is not enough.
- Keep the model forward contract: `forward(batch) -> {"logits": logits}`.
- Keep the standard batch keys unless the method truly needs a new dataset class.
- If a new batch key is required, document it in the paper method README and update the dataset code carefully.
- Do not add model-specific branches to `Data_pre.py` or `Dataset.py`. Add a focused adapter/Dataset under `reproflow/data/` or a paper-local `paper_methods/<method>/data.py`.

## Paper Method Folder

Create a folder:

```text
paper_methods/<method_name>/
├── method.yaml
├── README.md
└── reproduction_notes.md
```

`method.yaml` must state:

- paper title
- task type
- expected input fields
- output target
- required configs
- baseline methods
- validation commands

## Validation Checklist

Before claiming the reproduction is complete, run:

```bash
python scripts/doctor.py data=<dataset> model=<method> trainer=<trainer> metrics=default
python main.py data=<dataset> model=<method> trainer=<trainer> training_loop.epochs=1
python -m py_compile main.py Data_pre.py Dataset.py engine.py reproflow/reports.py
```

If the method changes a standard trainer, also run the sample smoke tests:

```bash
python main.py data=sample_binary trainer=binary training_loop.epochs=1
python main.py data=sample_multiclass trainer=multiclass training_loop.epochs=1
python main.py data=sample_regression trainer=regression training_loop.epochs=1
```

If the method introduces meaningful hyperparameters, add a tuning spec:

```text
configs/tuning/<method_name>_grid.yaml
```

Validate it with:

```bash
python scripts/tuning/run_grid_search.py configs/tuning/<method_name>_grid.yaml --dry-run --max-runs 2
```

If the method needs controlled component removal or variants, add an ablation spec:

```text
configs/ablation/<method_name>_ablation.yaml
```

Validate it with:

```bash
python scripts/ablation/run_ablation.py configs/ablation/<method_name>_ablation.yaml --dry-run --max-runs 2
```

For both tuning and ablation, use `seeds` as the standard multi-seed interface:

```yaml
seeds: [42, 43, 44]
seed_overrides:
  - random.seed
seed_controls_data_split: false
```

Set `seed_controls_data_split: true` only when the experiment design should vary the train/test split across seeds.

## Fair Comparison Manifest

For the final paper-method comparison, add or update:

```text
configs/experiment/<paper_or_task>_comparison.yaml
```

The manifest should include baseline methods and the paper method under the same dataset, metrics, and seed list:

```yaml
experiment_name: paper_xxx_comparison
data: <dataset>
metrics: default
seeds: [42, 43, 44]
methods:
  - name: traditional_ml
    type: ml_benchmark
    primary_metric: roc_auc
  - name: paper_xxx
    type: deep_learning
    model: paper_xxx
    trainer: binary
```

Validate it first:

```bash
python scripts/experiment/run_experiment.py configs/experiment/<paper_or_task>_comparison.yaml --dry-run --max-runs 2
```

## Scaffold Command

After putting a paper under `docs/papers/`, create the implementation scaffold with:

```bash
python scripts/paper_methods/scaffold.py <method_name> --paper docs/papers/<paper>.pdf --dataset <dataset> --trainer <trainer>
```

The scaffold creates `method.yaml`, an implementation checklist, a model config, a tuning spec, and an ablation spec. The generated model config is a placeholder until the AI agent implements the matching model class.

## Example Paper Library

ReproFlow includes 10 example-only classic paper scaffolds under:

```text
paper_methods/examples/
configs/model/examples/
configs/trainer/examples/
models/paper_example_models.py
```

Use these examples as patterns before adding a new paper method:

- Transformer encoder example
- BERT classifier example with an auxiliary head
- ResNet-style MLP example
- Wide & Deep example
- Neural Collaborative Filtering example
- DeepFM example
- Deep & Cross Network example
- AutoInt example
- Deep Interest Network example with an auxiliary head
- DLRM example

These files are marked `EXAMPLE ONLY`. They show file layout, model YAML, and optional trainer YAML. They are not faithful reproductions and should not be reported as experimental results.

For data contracts beyond plain tabular CSV, inspect:

- `configs/data/examples/recommender_pairwise_example.yaml`
- `configs/data/examples/graph_minibatch_example.yaml`
- `reproflow/data/recommender.py`
- `reproflow/data/graph.py`

## Output Requirement

Every reproduction should at least leave the default lightweight outputs:

- training log
- history CSV
- best checkpoint

For a final comparison or handoff, enable markdown report generation. The report should identify:

- dataset
- task type
- model config
- trainer config
- metrics
- history CSV
- resolved config snapshot
- best checkpoint
- experiment id
- run id

Enable richer artifacts only when needed:

```bash
python main.py report.enabled=true tracking.enabled=true artifacts.save_predictions=true artifacts.save_manifest=true
```

After running baselines, tuning, ablation, or multi-seed experiments, generate the aggregate report:

```bash
python scripts/reports/generate_experiment_report.py
```

## When To Add A New Trainer

Add a new trainer only for a real training-contract change:

- auxiliary loss
- ranking/listwise objective
- contrastive objective
- sequence generation objective
- graph-specific metrics or batching
- multi-task heads requiring custom loss aggregation

For ordinary supervised prediction, use the existing binary, multiclass, or regression trainer.
