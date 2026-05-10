# 示例论文方法库

这里的 10 个文件夹都是示例 scaffold，面向小白用户和 AI agent。

它们展示一篇经典论文如何映射到：

- `paper_methods/examples/<method>/method.yaml`
- `configs/model/examples/<method>.yaml`
- 可选的 `configs/trainer/examples/*.yaml`
- 共享示例代码 `models/paper_example_models.py`

不要把这些示例当作严格论文复现，也不要把它们当作实验结果。默认也不建议在本地 Mac 上完整训练这些示例。

| 示例 | 论文 | 模型配置 | Trainer |
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

AI 可以用下面命令检查配置契约：

```bash
python scripts/doctor.py data=sample_binary model=examples/deepfm_example trainer=binary metrics=default
```

完整训练这些示例不是默认 README 流程的一部分。
