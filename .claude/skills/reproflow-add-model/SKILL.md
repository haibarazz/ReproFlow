---
name: reproflow-add-model
description: 在遵守 ReproFlow 模型、配置、数据和 trainer 契约的前提下，添加新的深度学习模型或 baseline。
---

# ReproFlow 添加模型

当用户需要添加 baseline、论文方法或模型结构变体时，使用这个 skill。

## 写代码前先看

- 如果是论文复现，先读 `../reproflow-reproduce-paper/references/paper-workflow.md`。
- 如果需要确认模型、数据或 trainer 契约，读 `../reproflow-reproduce-paper/references/contracts.md`。
- 如果模型需要非标准 batch 或非标准 loss，读 `../reproflow-reproduce-paper/references/decision-guide.md`。

## 必要文件

- `models/<model_name>.py` 或 `paper_methods/<method_name>/model.py`
- `configs/model/<model_name>.yaml`
- 可选：`configs/tuning/<model_name>_grid.yaml`
- 可选：`configs/ablation/<model_name>_ablation.yaml`

## 模型构造函数

尽量让模型支持这些通用参数：

```python
def __init__(self, input_dim: int, task_type: str, num_classes: int = 1, ...):
    ...
```

## Forward 契约

```python
def forward(self, batch):
    features = batch["basic_features"]
    return {"logits": logits}
```

期望输出形状：

- 二分类：`(batch,)` 或 `(batch, 1)`
- 多分类：`(batch, num_classes)`
- 回归：`(batch,)` 或 `(batch, 1)`

## 验证

```bash
python scripts/doctor.py data=sample_binary model=<model_name> trainer=binary metrics=default
python main.py data=sample_binary model=<model_name> trainer=binary metrics=default training_loop.epochs=1
```

如果模型只适配某个任务，只运行对应任务的 sample 检查。
