# 经典 arXiv 示例

这个文件列出 ReproFlow 内置的 10 个示例级论文方法 scaffold。它们用于帮助小白用户和 AI 理解如何把论文方法拆成 model、config 和 trainer 文件。

这些示例不是完整论文复现，也不是实验结果。

| 示例 | arXiv 论文 | 在 ReproFlow 中展示的核心思路 |
| --- | --- | --- |
| `transformer_encoder_example` | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 对 feature tokens 做 self-attention |
| `bert_classifier_example` | [BERT](https://arxiv.org/abs/1810.04805) | CLS pooling 加辅助 head |
| `resnet_mlp_example` | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | 把 residual blocks 适配到 tabular features |
| `wide_deep_example` | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) | wide linear branch 加 deep branch |
| `ncf_example` | [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) | user/item interaction branches |
| `deepfm_example` | [DeepFM](https://arxiv.org/abs/1703.04247) | wide、FM 和 deep branches |
| `dcn_example` | [Deep & Cross Network](https://arxiv.org/abs/1708.05123) | explicit cross layers 加 deep branch |
| `autoint_example` | [AutoInt](https://arxiv.org/abs/1810.11921) | self-attentive feature interaction |
| `din_example` | [Deep Interest Network](https://arxiv.org/abs/1706.06978) | target-aware attention 加辅助 interest head |
| `dlrm_example` | [DLRM](https://arxiv.org/abs/1906.00091) | bottom MLP、feature interactions、top MLP |
