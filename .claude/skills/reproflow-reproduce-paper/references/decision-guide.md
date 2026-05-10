# 决策指南

当需要判断论文改动应该放在哪里时，读取这个 reference。

## 只新增模型的情况

满足这些条件时，通常只需要新增模型：

- 方法消费默认 `basic_features` batch
- loss 是普通二分类、多分类或回归 loss
- 指标已经支持
- 只是模型结构或超参数变化

需要的文件：

```text
models/<method_name>.py
configs/model/<method_name>.yaml
paper_methods/<method_name>/method.yaml
```

## 需要新增 Data Adapter 的情况

出现这些情况时，新增 data adapter：

- 模型需要 user/item pairs
- 模型需要 graph node/edge tensors
- 模型需要多个 sequence fields
- 模型需要 multimodal inputs
- 预处理必须构造论文专用样本

参考：

```text
reproflow/data/recommender.py
reproflow/data/graph.py
configs/data/examples/recommender_pairwise_example.yaml
configs/data/examples/graph_minibatch_example.yaml
```

## 需要新增 Trainer 的情况

出现这些情况时，新增 trainer：

- loss 有多个组成部分
- 论文训练辅助 head
- 论文使用 pairwise/listwise ranking
- 优化步骤不同于标准监督训练
- 评估需要任务专用 batch 处理

保持 trainer 聚焦，不要把它写成第二个 `engine.py`。

## 需要新增 Metric 的情况

出现这些情况时，新增 metric：

- 论文主指标缺失
- 公平对比要求 baseline 和论文方法使用同一指标

需要的文件：

```text
metrics/
configs/metrics/*.yaml
```

## 使用现有示例

10 个论文示例覆盖常见实现形态：

- 普通架构：Transformer、ResNet-style MLP
- 预训练风格分类器：BERT
- tabular recommender 架构：Wide & Deep、DeepFM、DCN、AutoInt、DLRM
- 推荐系统专用模型：NCF、DIN
- 辅助 head trainer 形态：BERT 和 DIN 示例

这些只是实现 scaffold，不是实验结果。
