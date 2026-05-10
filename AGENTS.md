# ReproFlow Agent Guide

这个仓库是一个面向 AI 辅助论文复现的可复现实验框架。AI agent 应该通过项目已有契约添加论文方法，不要写独立训练脚本。

## 核心架构

运行主干：

- `main.py`：Hydra 深度学习训练入口
- `Data_pre.py`：兼容旧入口，内部委托给配置指定的 data adapter
- `Dataset.py`：默认 tabular Dataset 的兼容 alias
- `reproflow/data/`：tabular、推荐系统、图数据和未来数据形态的 adapter / Dataset 契约
- `engine.py`：二分类、多分类和回归 trainer
- `run_ml_benchmark.py`：传统 ML baseline 入口

除非用户明确要求迁移，否则不要把项目替换成 Lightning、MLflow、W&B 或其他框架。

## 必要工作流

添加或修改模型前：

1. 读取 `docs/architecture.md`。
2. 如果任务是论文复现，从 `.claude/skills/reproflow-reproduce-paper/SKILL.md` 开始。
3. 只读取当前步骤需要的 skill reference。
4. 运行对应预检查：

```bash
python scripts/doctor.py data=<dataset> model=<model> trainer=<trainer> metrics=default
```

## 扩展规则

- 数据 schema 放在 `configs/data/*.yaml`。
- 指标放在 `configs/metrics/*.yaml` 和 `metrics/`。
- 模型超参数放在 `configs/model/*.yaml`。
- 标准任务 trainer 放在 `configs/trainer/*.yaml` 和 `engine.py`。
- 新的非标准 trainer 应该放在独立模块，不要继续扩展 `engine.py`。
- 新数据形态应该使用 `data.adapter` 和 `data.dataset`，不要继续扩展 `Data_pre.py` 或 `Dataset.py`。
- 公平对比实验放在 `configs/experiment/*.yaml`。
- 调参配置放在 `configs/tuning/*.yaml`。
- 消融配置放在 `configs/ablation/*.yaml`。
- 论文专用文档放在 `paper_methods/<method_name>/`。

不要在模型代码里硬编码数据列、label 名称、指标、seed 或文件路径。

## 项目内置 Skills

可复用 AI workflow 放在 `.claude/skills/`。它们是普通 Markdown 文件，即使宿主工具不会自动加载 Claude skills，也可以让任何 AI agent 直接读取。

按任务选择：

- 接入数据集：`.claude/skills/reproflow-onboard-dataset/SKILL.md`
- 复现论文：`.claude/skills/reproflow-reproduce-paper/SKILL.md`
- 添加模型：`.claude/skills/reproflow-add-model/SKILL.md`
- 添加指标：`.claude/skills/reproflow-add-metric/SKILL.md`
- 跑公平对比：`.claude/skills/reproflow-run-fair-experiment/SKILL.md`
- 排查运行错误：`.claude/skills/reproflow-debug-run/SKILL.md`

## 验证

普通改动至少运行：

```bash
python -m py_compile main.py Data_pre.py Dataset.py engine.py run_ml_benchmark.py
python scripts/doctor.py data=sample_binary model=transformer trainer=binary metrics=default training_loop.epochs=1
```

如果改动影响训练行为，还要分别检查二分类、多分类和回归 sample 任务。

默认训练结束后，优先查看输出中的 `training_*.log` 和 `history_*.csv` 路径。结构化 artifact 是可选项，只有任务明确需要时才开启。
