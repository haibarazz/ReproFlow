# ReproFlow Copilot Instructions

Follow `AGENTS.md` first. Keep the root architecture centered on `main.py`, `Data_pre.py`, `Dataset.py`, and `engine.py`.

Do not create standalone paper reproduction scripts. Add datasets, metrics, models, ablations, tuning, and fair experiments through the existing `configs/`, `metrics/`, `models/`, `paper_methods/`, and `scripts/` contracts.

Before suggesting a completed implementation, run or recommend:

```bash
python scripts/doctor.py data=<dataset> model=<model> trainer=<trainer> metrics=default
```
