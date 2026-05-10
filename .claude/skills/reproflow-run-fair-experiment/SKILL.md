---
name: reproflow-run-fair-experiment
description: 使用统一指标和随机种子创建或运行 baseline 与论文方法的公平对比实验。
---

# ReproFlow 公平对比实验

当用户需要比较 baseline、论文方法、调参结果或消融实验时，使用这个 skill。

## 实验 Manifest

创建或修改 `configs/experiment/<name>.yaml`：

```yaml
experiment_name: <name>
data: <dataset>
trainer: <trainer>
training_loop: default
metrics: default
seeds: [42, 43, 44]
monitor_metric: val_auc
monitor_mode: max
benchmark_metric: roc_auc
methods:
  - name: traditional_ml
    type: ml_benchmark
    primary_metric: roc_auc
  - name: paper_method
    type: deep_learning
    model: paper_method
```

## 运行

先 dry-run：

```bash
python scripts/experiment/run_experiment.py configs/experiment/<name>.yaml --dry-run --max-runs 2
```

再正式运行：

```bash
python scripts/experiment/run_experiment.py configs/experiment/<name>.yaml
```

生成汇总报告：

```bash
python scripts/reports/generate_experiment_report.py
```

## 公平性规则

- 使用同一个 dataset config。
- 使用同一个 metric config。
- 使用同一个 seed 列表。
- 除非实验目标就是研究 split 方差，否则保持 `seed_controls_data_split: false`。
- 报告 seed mean 和 std。
