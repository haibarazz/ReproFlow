# Paper Workflow

Use this reference for the first implementation pass.

## 1. Paper Intake

Extract only the information needed for implementation:

- method name and paper title
- task type
- required input fields
- prediction target
- model blocks
- loss function
- metrics used by the paper
- baselines that should be compared
- assumptions that differ from the target dataset

Record this mapping in `paper_methods/<method_name>/method.yaml` before writing code.

## 2. Pick The Closest Existing Path

Start from the smallest working ReproFlow path:

- normal supervised CSV task: default tabular adapter + binary/multiclass/regression trainer
- recommendation ranking: inspect `configs/data/examples/recommender_pairwise_example.yaml`
- graph-style batches: inspect `configs/data/examples/graph_minibatch_example.yaml`
- auxiliary heads or custom objectives: inspect `reproflow/example_trainers.py`
- architecture examples: inspect `models/paper_example_models.py`

The example paper library is for structure, not experimental claims.

## 3. Scaffold

```bash
python scripts/paper_methods/scaffold.py <method_name> \
  --paper docs/papers/<paper>.pdf \
  --dataset <dataset> \
  --trainer <trainer>
```

Then update the generated files instead of creating a parallel layout.

## 4. Implement In This Order

1. `paper_methods/<method_name>/method.yaml`
2. model class and `configs/model/<method_name>.yaml`
3. data adapter only if the batch contract changes
4. trainer only if the objective changes
5. tuning or ablation config if the paper has important variants
6. fair comparison manifest under `configs/experiment/`

## 5. Handoff Notes

Leave a short note in the paper method folder explaining:

- what was faithfully implemented
- what was simplified
- what was not run
- which command validates the contract
