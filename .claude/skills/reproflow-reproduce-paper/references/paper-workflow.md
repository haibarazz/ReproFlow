# 论文复现流程

第一次实现论文方法时读取这个 reference。

## 1. 论文信息提取

只提取实现真正需要的信息：

- 方法名和论文标题
- 任务类型
- 需要的输入字段
- 预测目标
- 模型模块
- loss function
- 论文使用的 metrics
- 需要对比的 baselines
- 原论文设定与目标数据集的差异

写代码前，先把这些映射记录到 `paper_methods/<method_name>/method.yaml`。

## 2. 选择最接近的现有路径

从最小可运行 ReproFlow 路径开始：

- 普通监督 CSV 任务：默认 tabular adapter + binary/multiclass/regression trainer
- 推荐系统排序：查看 `configs/data/examples/recommender_pairwise_example.yaml`
- 图数据 batch：查看 `configs/data/examples/graph_minibatch_example.yaml`
- 辅助 head 或自定义 objective：查看 `reproflow/example_trainers.py`
- 架构示例：查看 `models/paper_example_models.py`

示例论文库只用于展示结构，不是实验结论。

## 3. 生成骨架

```bash
python scripts/paper_methods/scaffold.py <method_name> \
  --paper docs/papers/<paper>.pdf \
  --dataset <dataset> \
  --trainer <trainer>
```

然后修改生成的文件，不要另起一套并行结构。

## 4. 推荐实现顺序

1. `paper_methods/<method_name>/method.yaml`
2. 模型类和 `configs/model/<method_name>.yaml`
3. 只有 batch 契约变化时才新增 data adapter
4. 只有 objective 变化时才新增 trainer
5. 如果论文有重要变体，新增 tuning 或 ablation 配置
6. 在 `configs/experiment/` 下新增公平对比 manifest

## 5. 交接说明

在论文方法文件夹里留下简短说明：

- 哪些部分忠实实现了论文
- 哪些部分做了简化
- 哪些命令没有运行
- 哪个命令可以验证当前契约
