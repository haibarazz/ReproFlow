<div align="center">

<img src="docs/assets/reproflow-poster.png" alt="ReproFlow 项目海报" width="100%" />

<h1>ReproFlow</h1>

<p><strong>面向 AI 辅助论文复现的可复现实验框架</strong></p>

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-blue" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c" />
  <img alt="Hydra" src="https://img.shields.io/badge/Config-Hydra-4f46e5" />
  <img alt="Status" src="https://img.shields.io/badge/Status-MVP-green" />
</p>

</div>

ReproFlow 帮助深度学习初学者和 AI 助手把论文方法复现到自己的数据集上，并在统一的数据、指标、随机种子和 baseline 下进行公平对比。

<div align="center">
  <a href="./docs/assets/reproflow-architecture-guide.svg"><strong>查看小白友好的项目架构说明</strong></a>
</div>

![ReproFlow 工作流](docs/assets/reproflow-workflow.svg)

## 项目定位

现在 AI 已经很会写模型代码，真实深度学习项目还需要管好数据 schema、特征列、label、metric、seed、baseline、消融实验和日志输出。

ReproFlow 提供一个轻量、可配置、适合 AI 接入的实验基座：

```text
论文方法 -> 规范实现 -> 接入自定义数据集 -> 跑 baseline -> 多 seed / 消融 -> 输出可复现实验结果
```

面向小白的架构说明见 [docs/assets/reproflow-architecture-guide.svg](./docs/assets/reproflow-architecture-guide.svg)。

## 核心能力

- 自定义数据集：通过 `configs/data/*.yaml` 声明数据路径、label、特征列和 split。
- 数据适配器：通过 `adapter` / `dataset` 配置支持 tabular、推荐系统、图数据等不同 batch 形态。
- 常见任务：支持二分类、多分类、回归。
- 指标配置：通过 `configs/metrics/*.yaml` 选择评估指标，训练和评估口径保持一致。
- 传统 ML baseline：保留 `run_ml_benchmark.py`，方便和深度学习方法公平对比。
- 多随机种子：`configs/tuning/`、`configs/ablation/`、`configs/experiment/` 都支持 `seeds: [...]`。
- 调参和消融：支持 grid/candidates tuning，也支持一等公民的 ablation runner。
- AI 复现规范：内置 `AGENTS.md`、`.claude/skills/`、`paper_methods/template/`。
- 简洁训练输出：默认只输出一个训练 log、一个 history CSV 和 best checkpoint。

## 工作流

![ReproFlow 架构](docs/assets/reproflow-architecture.svg)

```text
放入数据和论文 -> 选择或新增模型 -> 统一训练 -> baseline 对比 -> 多 seed / 消融 -> 汇总结果
```

更完整的项目结构和扩展规范见 [docs/architecture.md](./docs/architecture.md)。

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

生成示例数据：

```bash
python scripts/data/generate_synthetic_data.py
```

训练一个二分类模型：

```bash
python main.py data=sample_binary model=transformer trainer=binary metrics=default
```

训练多分类和回归：

```bash
python main.py data=sample_multiclass model=transformer trainer=multiclass metrics=default
python main.py data=sample_regression model=transformer trainer=regression metrics=default
```

## 默认输出

默认训练输出保持克制：

```text
result/<dataset>/<model>/training_<time>.log
result/<dataset>/<model>/history_<time>.csv
checkpoints/<dataset>/<model>.pth
```

也就是说，默认只有：

- 一个训练日志
- 一个 epoch-level history CSV
- 一个 best checkpoint

如果确实需要更详细的报告、预测文件或结构化 tracking，可以在命令行显式打开：

```bash
python main.py report.enabled=true artifacts.save_predictions=true artifacts.save_manifest=true
```

## 接入自己的数据集

把数据放进 `dataset/`，然后新增一个数据配置：

```yaml
# configs/data/my_task.yaml
name: my_task
path: dataset/my_task/data.csv
task_type: binary_classification
id_col: sample_id
label_col: label
numeric_cols: [age, price, count]
categorical_cols: [department, city]
text_cols: []
split:
  strategy: random
  train_ratio: 0.8
  random_state: 42
preprocess:
  scale_numeric: true
  encode_categorical: true

adapter:
  _target_: reproflow.data.tabular.TabularDataAdapter
dataset:
  _target_: reproflow.data.tabular.TabularDataset
```

训练前先检查：

```bash
python scripts/doctor.py data=my_task model=transformer trainer=binary metrics=default
```

## 让 AI 复现论文

把论文放进：

```text
docs/papers/
```

生成论文方法模板：

```bash
python scripts/paper_methods/scaffold.py paper_xxx \
  --paper docs/papers/paper_xxx.pdf \
  --dataset my_task \
  --trainer binary
```

论文方法接入规范：

- `AGENTS.md`
- `docs/ai_reproduction_guide.md`
- `.claude/skills/reproflow-reproduce-paper/SKILL.md`
- `paper_methods/template/`

模型接入的详细接口放在 [docs/ai_reproduction_guide.md](./docs/ai_reproduction_guide.md)。

给 AI 的推荐提示词：

```text
请使用 .claude/skills/reproflow-reproduce-paper/SKILL.md，
把 docs/papers/<paper>.pdf 复现成 ReproFlow 方法。

目标数据：configs/data/<dataset>.yaml
任务类型：binary / multiclass / regression
限制：先不要跑完整训练，只跑 doctor 和必要的 1 epoch smoke check。
要求：不要写独立训练脚本，模型、数据、trainer、metric 都接入 ReproFlow 配置体系。
```

## 经典论文示例

项目内置了 10 个经典 arXiv 论文的示例级模型 scaffold，覆盖 Transformer、BERT、ResNet、Wide & Deep、NCF、DeepFM、DCN、AutoInt、DIN 和 DLRM。论文清单见 [docs/papers/classic_arxiv_examples.md](./docs/papers/classic_arxiv_examples.md)。

这些示例位于：

```text
paper_methods/examples/
configs/model/examples/
configs/trainer/examples/
models/paper_example_models.py
```

它们用于展示“小白 + AI”如何把论文方法拆成 model、model yaml 和必要的 trainer yaml。所有示例都标注为 `EXAMPLE ONLY`，默认不建议在本地 Mac 上直接完整训练。

项目也提供了两个数据形态示例，方便 AI 理解什么时候需要新增 adapter 和 dataset：

```text
configs/data/examples/recommender_pairwise_example.yaml
configs/data/examples/graph_minibatch_example.yaml
```

## 传统 ML Baseline

在同一个数据配置上跑传统机器学习 baseline：

```bash
python run_ml_benchmark.py data=sample_binary
python run_ml_benchmark.py data=sample_regression
```

## 调参

调参配置位于 `configs/tuning/`：

```bash
python scripts/tuning/run_grid_search.py configs/tuning/transformer_binary_grid.yaml --dry-run
python scripts/tuning/run_grid_search.py configs/tuning/transformer_binary_grid.yaml --max-runs 1
```

示例：

```yaml
seeds: [42, 43, 44]
grid:
  model.hidden_dim: [32, 64]
  training_loop.learning_rate: [0.001, 0.0005]
```

## 消融实验

消融配置位于 `configs/ablation/`：

```bash
python scripts/ablation/run_ablation.py configs/ablation/transformer_binary_ablation.yaml --dry-run
```

示例：

```yaml
variants:
  - name: full
    overrides: {}
  - name: small_hidden
    overrides:
      model.hidden_dim: 32
seeds: [42, 43, 44]
```

## 公平对比实验

推荐用 `configs/experiment/*.yaml` 管理完整对比：

```bash
python scripts/experiment/run_experiment.py configs/experiment/binary_smoke.yaml --dry-run
python scripts/experiment/run_experiment.py configs/experiment/binary_smoke.yaml --max-runs 1
```

它可以把深度学习方法和传统 ML baseline 放在同一组实验里，并统一 dataset、metric、seed 和输出目录。

## AI 协作流程

项目内置了面向 AI 协作的 skills。它们相当于给 AI 的项目内操作手册，让 AI 按固定流程接入数据、复现论文、添加模型、补指标、跑公平对比和排查错误。

```text
.claude/skills/
├── reproflow-onboard-dataset
├── reproflow-reproduce-paper
├── reproflow-add-model
├── reproflow-add-metric
├── reproflow-run-fair-experiment
└── reproflow-debug-run
```

最常用的入口是：

- `reproflow-reproduce-paper`：把 `docs/papers/` 里的论文复现成可配置方法。
- `reproflow-onboard-dataset`：把自己的数据集接入 `configs/data/`。
- `reproflow-run-fair-experiment`：把 baseline、论文方法、多 seed 和消融放进同一个对比实验。
- `reproflow-debug-run`：训练或配置出错时，先跑 doctor，再看日志和 history CSV。

skills 采用渐进式组织：主 `SKILL.md` 只放最短流程，细节放在 `references/` 里。AI 只有在需要判断 model、data adapter、trainer、metric 或验证命令时才读取对应材料。

10 个经典论文示例可以作为新增模型时的参考。推荐系统和图数据的 adapter 示例可以作为新增数据形态时的参考。

## 快速自检

```bash
python scripts/doctor.py data=sample_binary model=transformer trainer=binary metrics=default training_loop.epochs=1
python main.py data=sample_binary model=transformer trainer=binary metrics=default training_loop.epochs=1
```

## 交流与贡献

欢迎提交 Issue、提出建议，或者补充新的模型、数据适配、指标、论文方法模板和实验配置。

如果你在使用过程中遇到问题，也可以通过邮箱联系我：

```text
2812156857@qq.com
```
