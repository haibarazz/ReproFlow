---
name: reproflow-add-metric
description: 在不修改训练主循环的前提下，为 ReproFlow 添加或配置评估指标。
---

# ReproFlow 添加指标

当用户需要新增评估指标或指标组合时，使用这个 skill。

## 指标放在哪里

- 二分类和多分类指标：`metrics/classification.py`
- 回归指标：`metrics/regression.py`
- 排序指标占位：`metrics/ranking.py`
- 注册表和默认指标：`metrics/registry.py`
- 用户可选配置：`configs/metrics/*.yaml`

## 规则

- 不要把指标代码写进 `engine.py`。
- 指标函数保持纯函数：输入是 scores、labels 和 metric names。
- 返回扁平的 `dict[str, float]`。
- 通过任务对应的 metric set 把指标名加入 `SUPPORTED_METRICS`。
- 如果希望用户可以选择该指标，新增或更新 `configs/metrics/*.yaml`。

## 验证

```bash
python scripts/doctor.py data=sample_binary model=transformer trainer=binary metrics=default
python main.py data=sample_binary model=transformer trainer=binary metrics=default training_loop.epochs=1
```

如果是任务专用指标，也要运行对应任务的 sample 命令。
