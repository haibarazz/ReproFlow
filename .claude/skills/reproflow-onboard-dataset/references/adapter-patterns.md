# Adapter Patterns

Use this reference when a dataset does not fit the default tabular path.

## Default Tabular Path

Use this when every sample can become one row of features plus one label:

```yaml
adapter:
  _target_: reproflow.data.tabular.TabularDataAdapter
dataset:
  _target_: reproflow.data.tabular.TabularDataset
```

The default adapter handles:

- numeric columns
- categorical columns
- text columns converted to TF-IDF features
- random train/test split
- label encoding for classification

## Recommender Pairwise Path

Use this when a paper needs user/item pairs or negative sampling:

```text
configs/data/examples/recommender_pairwise_example.yaml
reproflow/data/recommender.py
```

Expected batch shape:

```python
{
    "user_id": Tensor,
    "pos_item_id": Tensor,
    "neg_item_id": Tensor,
    "label": Tensor
}
```

Implement ID mapping, split strategy, and negative sampling inside the adapter.

## Graph Path

Use this when a paper needs node/edge tensors:

```text
configs/data/examples/graph_minibatch_example.yaml
reproflow/data/graph.py
```

Expected batch shape:

```python
{
    "node_features": Tensor,
    "edge_index": Tensor,
    "label": Tensor
}
```

Implement graph construction and batching inside the adapter.

## Paper-Local Path

If the preprocessing is specific to one paper, start with:

```text
paper_methods/<method_name>/data.py
```

Promote it to `reproflow/data/` only after it becomes reusable.
