# ReproFlow 契约

当需要新增文件或确认实现是否符合框架时，读取这个 reference。

## 论文方法文件夹

```text
paper_methods/<method_name>/
├── method.yaml
├── implementation_checklist.md
├── README.md
└── reproduction_notes.md
```

`method.yaml` 应该包含：

- 论文标题
- 任务类型
- 期望输入
- 目标 label
- 模型配置
- trainer 配置
- 需要对比的 baselines
- 验证命令

## 模型契约

推荐构造函数：

```python
def __init__(self, input_dim: int, task_type: str, num_classes: int = 1, ...):
    ...
```

Forward 契约：

```python
def forward(self, batch):
    return {"logits": logits}
```

输出形状：

- 二分类：`(batch,)` 或 `(batch, 1)`
- 多分类：`(batch, num_classes)`
- 回归：`(batch,)` 或 `(batch, 1)`

## 数据契约

默认 batch keys：

```python
{
    "basic_features": Tensor,
    "label": Tensor,
    "sample_id": optional string
}
```

普通 CSV/tabular/text-feature 任务使用：

```yaml
adapter:
  _target_: reproflow.data.tabular.TabularDataAdapter
dataset:
  _target_: reproflow.data.tabular.TabularDataset
```

如果需要新的 batch 形态，在 `reproflow/data/` 中创建聚焦的 adapter/Dataset，或者先放在 `paper_methods/<method_name>/data.py`。

## 配置契约

用户可调参数写进 YAML，不要硬编码到 Python：

```text
configs/data/<dataset>.yaml
configs/model/<method_name>.yaml
configs/trainer/<trainer>.yaml
configs/tuning/<method_name>_grid.yaml
configs/ablation/<method_name>_ablation.yaml
configs/experiment/<comparison>.yaml
```

## Trainer 契约

除非论文需要下面这些变化，否则使用标准 trainer：

- auxiliary loss
- pairwise/listwise ranking objective
- contrastive objective
- sequence generation objective
- graph-specific batch handling
- multi-task loss aggregation

自定义 trainer 尽量放在 `engine.py` 外部，例如 `reproflow/example_trainers.py` 或其他聚焦模块。
