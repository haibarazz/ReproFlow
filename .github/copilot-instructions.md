# ReproFlow Copilot Instructions

优先遵守 `AGENTS.md`。项目的训练入口围绕 `main.py`、`Data_pre.py`、`Dataset.py`、`engine.py` 和 `reproflow/data/` 组织。

不要为论文复现创建独立训练脚本。数据集、指标、模型、消融、调参和公平对比都应该接入已有契约：

- `configs/`
- `metrics/`
- `models/`
- `paper_methods/`
- `scripts/`

在建议“实现完成”之前，运行或建议用户运行：

```bash
python scripts/doctor.py data=<dataset> model=<model> trainer=<trainer> metrics=default
```
