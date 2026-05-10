---
name: reproflow-reproduce-paper
description: 在 ReproFlow 中复现机器学习论文，把 docs/papers 中的论文转成方法插件、配置、adapter/trainer 选择、smoke check 和公平对比实验。
---

# ReproFlow 论文复现

当用户要求 AI 复现论文、添加论文方法，或把论文方法迁移到自定义数据集时，使用这个 skill。

## 必要输入

- 论文文件或笔记位于 `docs/papers/`
- 目标数据集配置位于 `configs/data/`
- 任务类型和 trainer：`binary`、`multiclass`、`regression`，或有充分理由的自定义 trainer
- 是否允许完整训练；默认只跑 doctor 和极小 smoke run

## 渐进式读取

只读取当前步骤需要的内容：

- 第一次实现：`references/paper-workflow.md`
- 文件契约和 YAML 形状：`references/contracts.md`
- 需要判断 model / trainer / data adapter：`references/decision-guide.md`
- 准备交付前：`references/verification.md`
- 如果仍不清楚，再读 `docs/ai_reproduction_guide.md`

## 默认流程

1. 识别论文任务、输入、目标、loss 和评估指标。
2. 检查目标数据配置和 batch 形态。
3. 生成方法骨架：

```bash
python scripts/paper_methods/scaffold.py <method_name> --paper docs/papers/<paper>.pdf --dataset <dataset> --trainer <trainer>
```

4. 先填写 `paper_methods/<method_name>/method.yaml`，再写模型代码。
5. 用最小改动实现论文需要的 model/config/trainer。
6. 任何训练命令前先运行 doctor。
7. 把论文方法加入 `configs/experiment/` 下的公平对比 manifest。

## 不可破坏的契约

```python
def forward(self, batch):
    return {"logits": logits}
```

- 不要写独立训练脚本。
- 不要在模型代码里硬编码数据列或 label。
- 不要往 `Data_pre.py`、`Dataset.py` 或 `engine.py` 里添加模型专用分支。
- 只有 batch 形态改变时才新增 data adapter。
- 只有 loss、objective 或优化循环改变时才新增 trainer。
- 如果代码或配置只是 scaffold，要明确标记为示例，不要伪装成可运行复现。
