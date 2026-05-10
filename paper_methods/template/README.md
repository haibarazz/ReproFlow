# 论文方法模板

实现一篇论文方法前，把这个文件夹复制到：

```text
paper_methods/<method_name>/
```

## 必要文件

- `method.yaml`：论文到实现的结构化映射
- `README.md`：方法简介和运行命令
- `reproduction_notes.md`：实现决策、与原论文的差异和验证结果

## 模型契约

模型必须接收标准 ReproFlow batch，并返回：

```python
{"logits": logits}
```

只有当论文方法无法表达为普通监督预测时，才新增自定义 batch 字段或自定义 trainer。

## 预期运行命令

```bash
python main.py data=<dataset_name> model=<method_name> trainer=<trainer_name>
```
