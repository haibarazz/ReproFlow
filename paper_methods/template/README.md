# Paper Method Template

Copy this folder to `paper_methods/<method_name>/` before implementing a paper method.

## Required Files

- `method.yaml`: structured paper-to-implementation mapping
- `README.md`: method summary and run commands
- `reproduction_notes.md`: implementation decisions, deviations from the paper, and validation results

## Model Contract

The model must accept the standard ReproFlow batch and return:

```python
{"logits": logits}
```

Only add custom batch fields or a custom trainer when the paper method cannot be expressed as ordinary supervised prediction.

## Expected Run Command

```bash
python main.py data=<dataset_name> model=<method_name> trainer=<trainer_name>
```
