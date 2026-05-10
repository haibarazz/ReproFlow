# AI 论文复现指南

这份文档定义了 AI 编程助手在 ReproFlow 中复现论文时必须遵守的项目契约。

## 推荐入口

如果 AI 可以读取项目内置 skills，优先从这里开始：

```text
.claude/skills/reproflow-reproduce-paper/SKILL.md
```

这个 skill 会先给出最短流程，并在需要时再引导 AI 读取 `references/` 中的细节文件。本指南作为更完整的项目契约使用。

## 必要输入

1. 目标论文文件或笔记放在 `docs/papers/`。
2. 数据文件放在 `dataset/`。
3. 数据 schema 写在 `configs/data/<dataset_name>.yaml`。
4. 明确任务类型：`binary_classification`、`multiclass_classification` 或 `regression`。

## 实现规则

- 不要为论文方法写独立训练脚本。
- 不要在 Python 代码里硬编码特征列、label 名称或文件路径。
- 数据 schema 放在 `configs/data/*.yaml`。
- 当方法需要新的数据形态时，使用 `data.adapter` 和 `data.dataset`。
- 模型超参数和 `_target_` 放在 `configs/model/*.yaml`。
- 指标放在 `configs/metrics/*.yaml`，不要把指标代码塞进 `engine.py`。
- 只有现有 trainer 不够时，才新增 `configs/trainer/*.yaml` 和对应 trainer。
- 模型 forward 契约保持为：`forward(batch) -> {"logits": logits}`。
- 如果只是普通监督学习，优先使用标准 batch key。
- 如果必须新增 batch key，需要在论文方法 README 中说明，并谨慎更新数据代码。
- 不要往 `Data_pre.py` 或 `Dataset.py` 里添加模型专用分支。新增数据形态应该放在 `reproflow/data/`，或者先放在 `paper_methods/<method>/data.py`。

## 论文方法文件夹

每个论文方法都应该有独立文件夹：

```text
paper_methods/<method_name>/
├── method.yaml
├── README.md
└── reproduction_notes.md
```

`method.yaml` 至少说明：

- 论文标题
- 任务类型
- 输入字段
- 输出目标
- 需要的配置文件
- 需要对比的 baseline
- 验证命令

## 验证清单

在声称复现完成之前，至少运行：

```bash
python scripts/doctor.py data=<dataset> model=<method> trainer=<trainer> metrics=default
python main.py data=<dataset> model=<method> trainer=<trainer> training_loop.epochs=1
python -m py_compile main.py Data_pre.py Dataset.py engine.py reproflow/reports.py
```

如果方法修改了标准 trainer，还要跑对应 sample smoke tests：

```bash
python main.py data=sample_binary trainer=binary training_loop.epochs=1
python main.py data=sample_multiclass trainer=multiclass training_loop.epochs=1
python main.py data=sample_regression trainer=regression training_loop.epochs=1
```

如果方法引入了重要超参数，新增 tuning 配置：

```text
configs/tuning/<method_name>_grid.yaml
```

先用 dry-run 验证：

```bash
python scripts/tuning/run_grid_search.py configs/tuning/<method_name>_grid.yaml --dry-run --max-runs 2
```

如果方法需要组件移除或变体对比，新增 ablation 配置：

```text
configs/ablation/<method_name>_ablation.yaml
```

先用 dry-run 验证：

```bash
python scripts/ablation/run_ablation.py configs/ablation/<method_name>_ablation.yaml --dry-run --max-runs 2
```

tuning 和 ablation 都使用 `seeds` 作为标准多随机种子接口：

```yaml
seeds: [42, 43, 44]
seed_overrides:
  - random.seed
seed_controls_data_split: false
```

只有当实验目标就是研究 train/test split 方差时，才设置：

```yaml
seed_controls_data_split: true
```

## 公平对比实验

最终论文方法对比应该新增或更新：

```text
configs/experiment/<paper_or_task>_comparison.yaml
```

这个 manifest 应该让 baseline 和论文方法共享同一个数据集、指标和 seed 列表：

```yaml
experiment_name: paper_xxx_comparison
data: <dataset>
metrics: default
seeds: [42, 43, 44]
methods:
  - name: traditional_ml
    type: ml_benchmark
    primary_metric: roc_auc
  - name: paper_xxx
    type: deep_learning
    model: paper_xxx
    trainer: binary
```

先验证：

```bash
python scripts/experiment/run_experiment.py configs/experiment/<paper_or_task>_comparison.yaml --dry-run --max-runs 2
```

## 脚手架命令

把论文放到 `docs/papers/` 后，用下面命令生成方法骨架：

```bash
python scripts/paper_methods/scaffold.py <method_name> --paper docs/papers/<paper>.pdf --dataset <dataset> --trainer <trainer>
```

脚手架会生成 `method.yaml`、实现清单、模型配置、tuning 配置和 ablation 配置。生成的模型配置只是占位，AI 需要继续实现对应模型类。

## 经典论文示例库

ReproFlow 内置了 10 个经典论文的示例 scaffold：

```text
paper_methods/examples/
configs/model/examples/
configs/trainer/examples/
models/paper_example_models.py
```

新增论文方法前，可以参考这些示例：

- Transformer encoder example
- BERT classifier example，带辅助 head
- ResNet-style MLP example
- Wide & Deep example
- Neural Collaborative Filtering example
- DeepFM example
- Deep & Cross Network example
- AutoInt example
- Deep Interest Network example，带辅助 head
- DLRM example

这些文件都标记为 `EXAMPLE ONLY`。它们展示文件组织、模型 YAML 和可选 trainer YAML，不是严格复现结果，也不应该作为实验结果汇报。

如果数据契约超出普通 tabular CSV，可以参考：

- `configs/data/examples/recommender_pairwise_example.yaml`
- `configs/data/examples/graph_minibatch_example.yaml`
- `reproflow/data/recommender.py`
- `reproflow/data/graph.py`

## 输出要求

每次复现至少应该留下默认轻量输出：

- training log
- history CSV
- best checkpoint

最终对比或交接时，可以开启 markdown report。报告应该能说明：

- 数据集
- 任务类型
- 模型配置
- trainer 配置
- 指标配置
- history CSV
- resolved config snapshot
- best checkpoint
- experiment id
- run id

只有需要更多 artifact 时，才开启更详细输出：

```bash
python main.py report.enabled=true tracking.enabled=true artifacts.save_predictions=true artifacts.save_manifest=true
```

运行 baseline、tuning、ablation 或多 seed 实验后，可以生成汇总报告：

```bash
python scripts/reports/generate_experiment_report.py
```

## 什么时候新增 Trainer

只有训练契约真的改变时才新增 trainer，例如：

- auxiliary loss
- ranking/listwise objective
- contrastive objective
- sequence generation objective
- graph-specific metrics or batching
- multi-task heads requiring custom loss aggregation

普通监督预测任务使用现有二分类、多分类或回归 trainer。
