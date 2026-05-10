# Data Adapters

ReproFlow uses data adapters to keep data preparation from growing into one large
`Data_pre.py` file.

## Mental Model

```text
raw files
  -> DataAdapter
  -> Dataset
  -> DataLoader
  -> model/trainer
```

- `DataAdapter` replaces the growing responsibilities of `Data_pre.py`.
- `Dataset` replaces the growing responsibilities of `Dataset.py`.
- `configs/data/*.yaml` decides which adapter and Dataset are used.

## Default Tabular Adapter

```yaml
adapter:
  _target_: reproflow.data.tabular.TabularDataAdapter
dataset:
  _target_: reproflow.data.tabular.TabularDataset
```

Use this for ordinary CSV tasks with numeric, categorical, and text columns.

## When To Add A New Adapter

Add a new adapter when the model needs a new batch shape, such as:

- user/item pairwise ranking
- graph node/edge batches
- multi-turn sequence fields
- multimodal inputs
- paper-specific precomputed embeddings

Put reusable adapters in `reproflow/data/`. Put paper-specific adapters in
`paper_methods/<method>/data.py` until they prove reusable.

## Example-Only Configs

```text
configs/data/examples/recommender_pairwise_example.yaml
configs/data/examples/graph_minibatch_example.yaml
```

These are not runnable by default. They are examples for users and AI agents.
