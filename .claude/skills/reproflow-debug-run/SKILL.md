---
name: reproflow-debug-run
description: 使用配置、doctor 检查、日志和生成文件诊断 ReproFlow 运行失败。
---

# ReproFlow 排查运行错误

当训练、调参、消融、实验或 benchmark 运行失败时，使用这个 skill。

## 优先检查

```bash
python scripts/doctor.py data=<dataset> model=<model> trainer=<trainer> metrics=default
```

然后查看：

- `result/**/training_*.log`
- `result/**/config_*.yaml`
- `result/**/history_*.csv`
- `result/**/summary_latest.csv`

以下结构化 artifact 只有显式开启时才会出现：

- `result/tracking/<experiment_id>/<run_id>/run_metadata.json`
- `result/tracking/<experiment_id>/<run_id>/artifacts_manifest.json`
- `result/tracking/<experiment_id>/<run_id>/events.jsonl`
- `result/tracking/<experiment_id>/<run_id>/metrics_latest.json`
- `result/tracking/<experiment_id>/<run_id>/predictions.csv`

## 常见失败原因

- 数据配置引用了不存在的列。
- `task_type` 和 trainer 不匹配。
- `training_loop.monitor_metric` 没有包含在当前 metrics 配置中。
- 模型 forward 没有返回 `{"logits": logits}`。
- 多分类 logits 形状不是 `(batch, num_classes)`。
- 二分类或回归 logits 多了一维不兼容形状。

## 修复顺序

1. 先修配置错误。
2. 再修模型 forward 契约。
3. 只有方法目标真的需要时，才改 trainer。
4. 跑完整实验前先重新运行 doctor。
