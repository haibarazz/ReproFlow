# 数据适配器说明

ReproFlow 使用 data adapter 避免数据处理逻辑继续堆进一个巨大的 `Data_pre.py`。

## 核心理解

```text
原始文件
  -> DataAdapter
  -> Dataset
  -> DataLoader
  -> model/trainer
```

- `DataAdapter` 承担原来会让 `Data_pre.py` 变大的职责。
- `Dataset` 承担原来会让 `Dataset.py` 变大的职责。
- `configs/data/*.yaml` 决定使用哪个 adapter 和 Dataset。

## 默认 Tabular Adapter

```yaml
adapter:
  _target_: reproflow.data.tabular.TabularDataAdapter
dataset:
  _target_: reproflow.data.tabular.TabularDataset
```

普通 CSV 任务可以直接使用它，支持 numeric、categorical 和 text columns。

## 什么时候新增 Adapter

当模型需要新的 batch 形态时，新增 adapter，例如：

- user/item pairwise ranking
- graph node/edge batches
- multi-turn sequence fields
- multimodal inputs
- paper-specific precomputed embeddings

可复用 adapter 放在 `reproflow/data/`。只服务于某篇论文的 adapter 先放在 `paper_methods/<method>/data.py`，等它变得通用后再提升到 `reproflow/data/`。

## 示例配置

```text
configs/data/examples/recommender_pairwise_example.yaml
configs/data/examples/graph_minibatch_example.yaml
```

这些配置默认不可直接运行，只作为用户和 AI 理解扩展方式的示例。
