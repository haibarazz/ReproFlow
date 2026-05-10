---
name: reproflow-onboard-dataset
description: 通过数据 YAML、adapter/Dataset 契约和 doctor 检查，把自定义数据集接入 ReproFlow。
---

# ReproFlow 接入数据集

当用户想把自己的数据集接入 ReproFlow 时，使用这个 skill。

## 渐进式读取

- 普通 CSV/tabular 数据集：直接走下面的快速路径。
- 新 batch 形态或论文专用预处理：读取 `references/adapter-patterns.md`。
- 如果是在论文复现任务中接入数据，也读取 `../reproflow-reproduce-paper/references/contracts.md`。

## 快速路径

1. 把数据文件放到 `dataset/<task_name>/` 或 `dataset/<task_name>.csv`。
2. 创建 `configs/data/<task_name>.yaml`。
3. 声明数据路径、label、特征列、split、preprocess、`adapter` 和 `dataset`。
4. 普通 tabular 或 text-feature 数据集使用：

```yaml
adapter:
  _target_: reproflow.data.tabular.TabularDataAdapter
dataset:
  _target_: reproflow.data.tabular.TabularDataset
```

5. 推荐系统、图数据、序列数据或论文专用 batch 形态，需要在 `reproflow/data/` 或 `paper_methods/<method>/data.py` 中新增聚焦的 adapter/Dataset。
6. 普通数据接入不要修改 `Data_pre.py` 或 `Dataset.py`。
7. 验证契约：

```bash
python scripts/doctor.py data=<task_name> model=transformer trainer=<binary|multiclass|regression> metrics=default
```

## 常见修复

- 如果某列缺失，修 `configs/data/<task_name>.yaml`，不要改模型代码。
- 如果 label 是字符串，让预处理流程自动编码。
- 如果二分类出现超过两个 label，改成 `multiclass_classification` 或修正 label 定义。
- 做模型 seed 对比时，默认使用 `seed_controls_data_split: false`，除非实验明确研究 split 方差。
- 当方法需要不同 batch 契约时，参考 `configs/data/examples/recommender_pairwise_example.yaml` 和 `configs/data/examples/graph_minibatch_example.yaml`。
