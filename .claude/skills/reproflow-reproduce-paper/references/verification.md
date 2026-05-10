# 验证

在声称论文方法可用之前，读取这个 reference。

## 先做轻量检查

```bash
python scripts/doctor.py data=<dataset> model=<method_name> trainer=<trainer> metrics=default
python -m py_compile main.py Data_pre.py Dataset.py engine.py
```

如果论文方法新增了文件，也编译对应文件：

```bash
python -m py_compile models/<method_name>.py
```

## 极小 Smoke Run

只有本地资源允许时才运行：

```bash
python main.py data=<dataset> model=<method_name> trainer=<trainer> metrics=default training_loop.epochs=1
```

如果本地机器资源有限，可以跳过完整训练，但必须说明已经运行了哪些检查。

## 如果修改了核心契约

运行对应 sample doctor 检查：

```bash
python scripts/doctor.py data=sample_binary model=transformer trainer=binary metrics=default training_loop.epochs=1
python scripts/doctor.py data=sample_multiclass model=transformer trainer=multiclass metrics=default training_loop.epochs=1
python scripts/doctor.py data=sample_regression model=transformer trainer=regression metrics=default training_loop.epochs=1
```

## 调参、消融和对比实验

正式运行前先 dry-run：

```bash
python scripts/tuning/run_grid_search.py configs/tuning/<method_name>_grid.yaml --dry-run --max-runs 2
python scripts/ablation/run_ablation.py configs/ablation/<method_name>_ablation.yaml --dry-run --max-runs 2
python scripts/experiment/run_experiment.py configs/experiment/<comparison>.yaml --dry-run --max-runs 2
```

## 最终交接

交接时说明：

- 改了哪些文件
- 论文方法文件夹位置
- model / trainer / data adapter 选择
- 已运行的验证命令
- 明确没有运行的命令
