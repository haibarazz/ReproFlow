# Decision Guide

Use this reference when deciding where a paper change belongs.

## Add Only A Model When

- the method consumes the default `basic_features` batch
- the loss is ordinary binary, multiclass, or regression loss
- metrics are already supported
- only architecture or hyperparameters change

Files:

```text
models/<method_name>.py
configs/model/<method_name>.yaml
paper_methods/<method_name>/method.yaml
```

## Add A Data Adapter When

- the model needs user/item pairs
- the model needs graph node/edge tensors
- the model needs multiple sequence fields
- the model needs multimodal inputs
- preprocessing must create paper-specific derived examples

Start from:

```text
reproflow/data/recommender.py
reproflow/data/graph.py
configs/data/examples/recommender_pairwise_example.yaml
configs/data/examples/graph_minibatch_example.yaml
```

## Add A Trainer When

- the loss has multiple terms
- the paper trains auxiliary heads
- the paper uses pairwise/listwise ranking
- the optimization step differs from standard supervised training
- evaluation needs task-specific batch handling

Keep the trainer focused. Do not turn it into a second `engine.py`.

## Add A Metric When

- the paper's primary metric is missing
- a fair comparison requires the same metric across baseline and paper method

Files:

```text
metrics/
configs/metrics/*.yaml
```

## Use Existing Examples

The 10 paper examples cover common implementation shapes:

- plain architecture: Transformer, ResNet-style MLP
- pretrained-style classifier: BERT
- tabular recommender architecture: Wide & Deep, DeepFM, DCN, AutoInt, DLRM
- recommendation-specific model: NCF, DIN
- auxiliary-head trainer shape: BERT and DIN examples

They are implementation scaffolds only.
