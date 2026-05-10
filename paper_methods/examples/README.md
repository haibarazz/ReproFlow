# Example Paper Method Library

These 10 folders are example-only scaffolds for beginner users and AI agents.
They show how a classic paper can map into:

- `paper_methods/examples/<method>/method.yaml`
- `configs/model/examples/<method>.yaml`
- optional `configs/trainer/examples/*.yaml`
- shared example code in `models/paper_example_models.py`

Do not treat these examples as faithful reproductions or reported results. They
are not meant to be trained on a Mac by default.

| Example | Paper | Model config | Trainer |
| --- | --- | --- | --- |
| `transformer_encoder_example` | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | `configs/model/examples/transformer_encoder_example.yaml` | `trainer=binary` |
| `bert_classifier_example` | [BERT](https://arxiv.org/abs/1810.04805) | `configs/model/examples/bert_classifier_example.yaml` | `trainer=examples/auxiliary_binary` |
| `resnet_mlp_example` | [Deep Residual Learning](https://arxiv.org/abs/1512.03385) | `configs/model/examples/resnet_mlp_example.yaml` | `trainer=binary` |
| `wide_deep_example` | [Wide & Deep Learning](https://arxiv.org/abs/1606.07792) | `configs/model/examples/wide_deep_example.yaml` | `trainer=binary` |
| `ncf_example` | [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) | `configs/model/examples/ncf_example.yaml` | `trainer=examples/pairwise_ranking_example` |
| `deepfm_example` | [DeepFM](https://arxiv.org/abs/1703.04247) | `configs/model/examples/deepfm_example.yaml` | `trainer=binary` |
| `dcn_example` | [Deep & Cross Network](https://arxiv.org/abs/1708.05123) | `configs/model/examples/dcn_example.yaml` | `trainer=binary` |
| `autoint_example` | [AutoInt](https://arxiv.org/abs/1810.11921) | `configs/model/examples/autoint_example.yaml` | `trainer=binary` |
| `din_example` | [Deep Interest Network](https://arxiv.org/abs/1706.06978) | `configs/model/examples/din_example.yaml` | `trainer=examples/auxiliary_binary` |
| `dlrm_example` | [DLRM](https://arxiv.org/abs/1906.00091) | `configs/model/examples/dlrm_example.yaml` | `trainer=binary` |

Example dry-run style command for an AI agent to inspect the config contract:

```bash
python scripts/doctor.py data=sample_binary model=examples/deepfm_example trainer=binary metrics=default
```

Running full training for these examples is intentionally not part of the default
README flow.
