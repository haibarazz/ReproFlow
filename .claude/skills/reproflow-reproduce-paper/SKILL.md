---
name: reproflow-reproduce-paper
description: Reproduce an ML paper inside ReproFlow by turning docs/papers content into a method plugin, configs, adapter/trainer choices, smoke checks, and fair comparison manifests.
---

# ReproFlow Paper Reproduction

Use this skill when the user asks an AI agent to reproduce a paper, add a paper method, or adapt a paper method to a custom dataset in ReproFlow.

## Required Inputs

- paper file or notes under `docs/papers/`
- target dataset config under `configs/data/`
- task type and trainer: `binary`, `multiclass`, `regression`, or a justified custom trainer
- whether full training is allowed; default to doctor checks and tiny smoke runs

## Progressive Disclosure

Read only what the current step needs:

- First implementation pass: `references/paper-workflow.md`
- File contracts and YAML shapes: `references/contracts.md`
- Need to decide model vs trainer vs data adapter: `references/decision-guide.md`
- Before claiming done: `references/verification.md`
- If still unclear, read `docs/ai_reproduction_guide.md`

## Default Workflow

1. Identify the paper's task, inputs, target, loss, and evaluation metric.
2. Check the target data config and batch shape.
3. Scaffold the method:

```bash
python scripts/paper_methods/scaffold.py <method_name> --paper docs/papers/<paper>.pdf --dataset <dataset> --trainer <trainer>
```

4. Fill `paper_methods/<method_name>/method.yaml` before writing model code.
5. Implement the smallest model/config/trainer changes needed by the paper.
6. Run doctor before any training command.
7. Add the paper method to a fair comparison manifest under `configs/experiment/`.

## Non-Negotiable Contracts

```python
def forward(self, batch):
    return {"logits": logits}
```

- Do not write standalone training scripts.
- Do not hard-code dataset columns or labels in model code.
- Do not add model-specific branches to `Data_pre.py`, `Dataset.py`, or `engine.py`.
- Add a new data adapter only when the batch shape changes.
- Add a new trainer only when the loss, objective, or optimization loop changes.
- Mark example-only code and configs clearly if they are scaffolds rather than runnable implementations.
