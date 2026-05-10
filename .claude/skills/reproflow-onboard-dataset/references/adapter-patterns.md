# Adapter 模式

当数据集不适合默认 tabular 路径时，读取这个 reference。

## 默认 Tabular 路径

如果每个样本都能表示成一行 features 加一个 label，就使用：

```yaml
adapter:
  _target_: reproflow.data.tabular.TabularDataAdapter
dataset:
  _target_: reproflow.data.tabular.TabularDataset
```

默认 adapter 处理：

- numeric columns
- categorical columns
- text columns 转成 TF-IDF features
- random train/test split
- 分类 label encoding

## 推荐系统 Pairwise 路径

当论文需要 user/item pairs 或 negative sampling 时，参考：

```text
configs/data/examples/recommender_pairwise_example.yaml
reproflow/data/recommender.py
```

期望 batch 形态：

```python
{
    "user_id": Tensor,
    "pos_item_id": Tensor,
    "neg_item_id": Tensor,
    "label": Tensor
}
```

ID 映射、split 策略和 negative sampling 应该在 adapter 里实现。

## 图数据路径

当论文需要 node/edge tensors 时，参考：

```text
configs/data/examples/graph_minibatch_example.yaml
reproflow/data/graph.py
```

期望 batch 形态：

```python
{
    "node_features": Tensor,
    "edge_index": Tensor,
    "label": Tensor
}
```

图构建和 batching 应该在 adapter 里实现。

## 论文专用路径

如果预处理逻辑只服务于一篇论文，先放在：

```text
paper_methods/<method_name>/data.py
```

只有当它变得可复用时，再提升到 `reproflow/data/`。
