---
name: reproflow-reproduce-paper
description: Convert a paper under docs/papers into a ReproFlow paper method with configs, smoke checks, and reports.
---

# ReproFlow Paper Reproduction

Use this skill when implementing a method from a paper.

## Required Inputs

- Paper file under `docs/papers/`
- Target dataset config under `configs/data/`
- Task type: `binary_classification`, `multiclass_classification`, or `regression`

## Workflow

1. Read `docs/ai_reproduction_guide.md`.
2. Scaffold the method:

```bash
python scripts/paper_methods/scaffold.py <method_name> --paper docs/papers/<paper>.pdf --dataset <dataset> --trainer <trainer>
```

3. Fill `paper_methods/<method_name>/method.yaml`.
4. Implement model code under `models/` or `paper_methods/<method_name>/`.
5. Keep forward contract:

```python
def forward(self, batch):
    return {"logits": logits}
```

6. Validate:

```bash
python scripts/doctor.py data=<dataset> model=<method_name> trainer=<trainer> metrics=default
python main.py data=<dataset> model=<method_name> trainer=<trainer> metrics=default training_loop.epochs=1
```

7. Add the paper method to a fair comparison manifest under `configs/experiment/`.

## Rules

- Do not write standalone training scripts.
- Do not hard-code dataset columns or labels in model code.
- Do not add a new trainer unless the paper has a genuinely different loss, objective, or batch contract.
