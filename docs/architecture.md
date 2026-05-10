# ReproFlow 架构说明

ReproFlow 围绕数据、模型、训练器、指标、实验和论文方法建立明确接口。目标是让 AI 可以按规范增量添加新方法，而不是每复现一篇论文就写一堆零散脚本。

## AI 协作层

`AGENTS.md` 是给 AI 编程助手看的总入口。项目内置的 workflow skills 放在 `.claude/skills/`，用于描述可重复执行的任务流程：

- 接入新数据集
- 复现论文方法
- 添加新模型
- 添加新指标
- 组织公平对比实验
- 排查训练和配置错误

这些 skills 采用渐进式披露：每个 `SKILL.md` 只放最短流程，进一步细节放在同级 `references/` 文件中。AI 只有在需要判断模型、数据适配器、训练器或验证命令时，才读取对应 reference。

## 运行流程

```text
configs/data/*.yaml
        |
        v
Data_pre.py
  - 兼容旧入口
  - 调用 build_data(cfg)
        |
        v
reproflow/data/
  - TabularDataAdapter
  - 可扩展推荐系统 / 图数据 / 论文专用 adapter
  - Dataset 返回模型可消费的 batch
        |
        v
models/
  - forward(batch) -> {"logits": tensor}
        |
        v
engine.py
  - BinaryClassificationTrainer
  - MulticlassClassificationTrainer
  - RegressionTrainer
        |
        v
evaluators/ + metrics/
  - 按任务类型选择指标
  - binary / multiclass / regression 指标
        |
        v
reproflow/reports.py
  - history csv
  - config snapshot
  - best checkpoint
  - markdown report
        |
        v
reproflow/tracking.py
  - 本地 experiment_id / run_id 元信息
        |
        v
scripts/tuning/run_grid_search.py / scripts/ablation/run_ablation.py / scripts/experiment/run_experiment.py
  - 调参、候选配置、消融、公平对比、多 seed 运行
        |
        v
scripts/reports/generate_experiment_report.py
  - 汇总 baseline / tuning / ablation / seed 结果
```

## 扩展点

### 数据集

普通 tabular 或 text-feature 数据集只需要新增：

```text
configs/data/<name>.yaml
dataset/<name>.csv
```

不需要修改 Python 文件。默认 tabular 数据路径是：

```yaml
adapter:
  _target_: reproflow.data.tabular.TabularDataAdapter
dataset:
  _target_: reproflow.data.tabular.TabularDataset
```

如果论文或模型需要新的 batch 形态，就在 `reproflow/data/` 里新增小而聚焦的 adapter 和 Dataset：

```text
reproflow/data/tabular.py
reproflow/data/recommender.py
reproflow/data/graph.py
```

如果某个数据处理逻辑只服务于一篇论文，可以先放在：

```text
paper_methods/<paper>/data.py
```

只有当它变得通用时，再提升到 `reproflow/data/`。不要继续往 `Data_pre.py` 或 `Dataset.py` 里添加模型专用分支。

两个示例数据配置展示了这种扩展方式：

- `configs/data/examples/recommender_pairwise_example.yaml`
- `configs/data/examples/graph_minibatch_example.yaml`

### 模型

新增模型时，优先让模型类支持这些参数：

```python
def __init__(self, input_dim: int, task_type: str, num_classes: int = 1, ...):
    ...
```

模型的 `forward` 必须返回：

```python
{"logits": logits}
```

### 训练器

二分类、多分类和回归默认使用现有 trainer。只有当论文方法真的改变训练契约时，才新增 trainer，例如：

- 辅助 loss
- pairwise/listwise ranking objective
- contrastive objective
- sequence generation objective
- graph-specific batch behavior
- 多任务 loss 聚合

新增 trainer 应该放在独立聚焦的模块里，不要继续把 `engine.py` 堆大。

### 指标与评估

指标放在 `metrics/`，通过 `configs/metrics/*.yaml` 选择。trainer 不应该继续增长任务专用指标代码。trainer 把 scores 和 labels 交给 `evaluators/Evaluator`，再由 metric registry 分发计算。

V1 支持的指标族：

- 二分类：accuracy、precision、recall、F1、MCC、ROC-AUC、PR-AUC、Brier score
- 多分类：accuracy、balanced accuracy、macro/weighted precision/recall/F1、OVR/OVO AUC
- 回归：MSE、RMSE、MAE、MAPE、median AE、R2、explained variance
- 排序占位：HitRate@K、NDCG@K，后续给 ranking trainer 使用

### 实验

公平对比实验放在：

```text
configs/experiment/*.yaml
```

一个 experiment manifest 应该把 baseline、论文方法、数据集、指标、seed 和输出约定放在同一个配置里。深度学习方法通过 `main.py` 运行，传统 ML baseline 通过 `run_ml_benchmark.py` 运行。

### 调参

深度学习调参配置放在：

```text
configs/tuning/*.yaml
```

常用字段：

- `choices`：Hydra config group，例如 `data`、`model`、`trainer`、`training_loop`
- `base_overrides`：每次运行都会使用的固定 override
- `grid`：参数网格
- `candidates`：显式候选配置，适合网格太大时使用
- `seeds`：一等公民的随机种子列表
- `seed_overrides`：seed 写入哪些配置键，通常是 `random.seed`
- `monitor_metric` / `monitor_mode`：用于排序和选择最优结果

### 消融

消融配置放在：

```text
configs/ablation/*.yaml
```

消融使用命名 `variants`，不是宽泛网格。每个 variant 可以在同一组 `seeds` 上运行，runner 会输出单次结果和 seed mean/std 汇总。

### 输出与可选 tracking

默认训练输出保持轻量：

- `training_*.log`
- `history_*.csv`
- best checkpoint

结构化 tracking 是可选项，只在确实需要更多 artifact 时开启：

```bash
python main.py tracking.enabled=true report.enabled=true artifacts.save_manifest=true
```

可选 artifact 包括 markdown report、predictions CSV、events JSONL、metrics JSON 和 artifact manifest。

### Doctor

`scripts/doctor.py` 是给小白和 AI 用的预检查工具。它会检查：

- 数据文件和 schema 列是否存在
- task type 是否和标准 trainer 匹配
- 配置指标是否支持
- 模型能否实例化
- 模型 forward 是否返回 `{"logits": logits}`
- paper method 文件夹是否有必要元信息

### 论文方法

每个论文方法都应该从 `paper_methods/template/` 开始，让 AI 有固定位置记录论文映射、实现文件、配置和验证命令。

可以用脚手架命令生成初稿：

```bash
python scripts/paper_methods/scaffold.py <method_name> --paper docs/papers/<paper>.pdf --dataset <dataset> --trainer <trainer>
```

脚手架会生成：

- `paper_methods/<method>/method.yaml`
- `paper_methods/<method>/implementation_checklist.md`
- `configs/model/<method>.yaml`
- `configs/tuning/<method>_grid.yaml`
- `configs/ablation/<method>_ablation.yaml`

## 设计边界

ReproFlow 是实验基座，不是巨型深度学习平台。MLflow、Lightning、W&B、分布式训练、自动 PDF agent 都可以作为后续能力，但只有当它们真正服务于论文复现和公平实验时才应该引入。
