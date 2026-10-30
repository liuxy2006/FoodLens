# 16 图 Schema 学习验证

入口：[member_c_schema_overfit.ipynb](member_c_schema_overfit.ipynb)。仍在 `feat/member-c-diet-record`，审核推送后填写本次最终完整 commit SHA。
本地只进行代码检查和 CPU 测试，不下载权重或运行模型。Notebook 延用已有 Drive ZIP 准备、CUDA 保护与固定 SmolVLM revision。

本次本地实际执行 `python -m unittest discover -s tests`，81 项测试通过；两份 Notebook 的 nbformat 与全部代码单元语法校验通过。曲线渲染测试只使用人工夹具，不是模型成绩。
检查时本地 `data/raw/annotations/train.jsonl` 不存在，未独立复核这次真实 16 条标注；真实数据与监督 token 的核对会在 Colab 数据准备及训练检查中执行。新一轮 25/100 步尚未运行。

## 已知证据与检查结论

用户报告：commit `8b06fd88d66e9b6baa3195a2a936090946beaaba` 在 Colab T4 完成推理、LoRA 反向传播、120 个参数张量更新、1→2 步恢复，以及保存重载后的确定性生成一致。
用户提供的 step1/2 loss 为 1.75534558/1.64216316，val loss 为 1.66751886/1.65085125；两张验证图 JSON 可解析但 Schema 合法率 0。
上述数字来自用户实际运行报告，本地没有重新执行或独立核验 Drive 文件。不能据此声称模型已经学会 Schema 或营养估计。

| 检查 | 本次代码结论 |
|---|---|
| 训练与推理提示 | 均来自 interfaces.user_messages；没有两套不同提示。此次明确精确字段、items 的嵌套类型和数字类型，两端同步修改 |
| 图片输入 | collator 对完整对话和提示分别传入同一 images，推理同样用 RGB 图片；没有 dish_id 查表 |
| target_text | DietDataset 先验证 target_text 与 target JSON 一致，再用 target 的规范 JSON 作监督文本；不会引入错误字段映射 |
| 损失掩码 | 提示 token 必须为完整序列的严格前缀，提示和 padding 标签设为 -100；未发现明确的边界错误 |
| 图像 token | 位于提示区，被屏蔽；verify-update 额外检查实际 pixel_values、图像 token 和其标签 |
| 目标截断 | 原来超过 max_length 会报错，没有发现静默截断；现显式 truncation=False，并解码被监督 token 与 target 比较 |
| 聊天模板 | 训练 prompt 与推理均 add_generation_prompt=True，训练完整文本包含 assistant 答案；严格前缀检查约束二者 |
| LoRA | 维持 text_model 内 q_proj/v_proj，rank=8、alpha=16、lr=1e-4，不改视觉编码器或扩展注入范围 |
| Processor | 基础 Processor 与模型同版本，恢复/推理时从 Adapter 目录加载保存的 Processor |

未发现足以解释错误字段的明确标签映射 bug。两步 smoke 的目的只验证软件链路，不足以判断格式学习能力。
本次还修正了梯度累积时 `steps.jsonl.loss` 只记录最后一个 microbatch 的问题：现在记录该更新组的平均 loss，并单列最后一个 microbatch loss。默认累积 1 的小实验不受旧统计问题影响。
新增中途评价会恢复 model 原有 train/eval 模式；不会使后续训练一直处于 eval。

## 实验设计与操作

1. 在新 Colab T4 运行时打开独立 Notebook，按顺序运行代码/依赖/Drive/数据准备单元。模型仍为 SmolVLM-256M，revision 不变。
2. 从原始预训练模型新建 LoRA。固定 train 的前 16 张真实图片，val 的前 4 张固定用于监控，test 不进入实验评价或训练。
3. `overfit_config.json` 为 batch 1、累积 1、7 epochs：`7 × ceil(16/1) = 112` 个可用优化器步骤，首次入口硬停在 25 步。
4. 检查步骤 0/25 的生成报告、每步 loss 曲线、监督文本和原文。模型未生成 JSON或命中 token 上限时必须查看原文，不进行字段重命名/答案补全。
5. 人工填写 `REVIEW_NOTE` 后才打开 `CONTINUE_TO_100=True`，从本实验断点恢复至**总计 100 步**。不允许 fresh 100 步入口，不接受 smoke/full-data 断点。
6. 100 步训练子集 Schema 仍为 0 时，`decision.json` 给出 STOP_AND_DIAGNOSE；不要增加 epoch、提高上限或先跑全量数据。

独立命令（在 Colab 的仓库根目录执行）：

```bash
python -m task2_diet_record.schema_experiment --model-path <模型目录> --revision <固定完整模型SHA> \
  --data-root /content/foodlens_data --stage 25 --output <新的Drive实验目录>/schema_step25

python -m task2_diet_record.schema_experiment --model-path <同一模型目录> --revision <同一模型SHA> \
  --data-root /content/foodlens_data --stage 100 \
  --resume <schema_step25/last_checkpoint.json指向的完整断点目录> \
  --output <新的Drive实验目录>/schema_step100
```

断线时可恢复本实验较早的完整断点，但步数上限仍是 25 或 100；数据、配置、提示词哈希和监控协议不匹配会被拒绝。
生成评价使用 max_new_tokens=1024、do_sample=False，记录生成 token 数和是否到达上限；不设置约束解码或修复输出，便于观察模型本身是否学会字段。
25 步和 100 步分别用新目录。Notebook 根目录为 `FoodLens/outputs/schema_overfit_<运行ID>/`，不会改写旧 pretrained_inference/smoke_step1/smoke_resumed/val_metrics。

## 要查看的结果

- `run.json`：实际源码版本、提示全文及哈希、聊天模板、模型/数据签名、LoRA 参数数量、固定生成样本 ID。
- `batch_inspection_epoch_*.json`：实际图像 token/像素形状、掩码统计、解码后的监督答案以及 matches_target。
- `steps.jsonl`：每个优化器更新的 loss、有限/非零梯度、参数更新、实际耗时和显存峰值。不是全程时间估计。
- `monitor/step-*.json`：步骤 0、25、50、75、100（按实际运行阶段产生）的验证损失、train/val 原始输出和全部指标。
- `predictions[].schema`：items 是否为数组、缺少/多出的字段、哪些数值是数字类型、Schema 具体错误。错误字段统计不会悄悄修复生成文本。
- `decision.json`：停止/复核提示，不自动启动正式训练。
- `curves25/`、`curves100/`：PNG 学习曲线、监控点 CSV 和逐步训练 loss CSV。
- `checkpoint_times.jsonl` 与监控报告的 `monitor_seconds`：真实保存和验证/生成的耗时；metrics.jsonl 另记包含模型加载、验证、保存的累计运行时间，用于区分单步计算与端到端开销。

固定 4 张 val 的 teacher-forcing loss 用于快速诊断，不能冒充完整 414 张验证集结果。
训练子集生成分数只反映记忆能力，不是泛化成绩。Schema 合法率也是有效覆盖率；另报 JSON 解析率、micro/macro F1、五项合法预测 MSE 和全样本零补值 MSE，并保留各自分母。
没有合法预测时，有效 F1/MSE 为 null，图上留空，不画成零误差。未匹配原料重量只作诊断。

## 何时考虑完整训练

应同时看到：训练子集 Schema 接近全对（例如至少 15/16）、字段/数字类型稳定，验证子集不再全为非法，监督解码核对通过，没有 NaN/Inf 或异常截断。
还应检查原料 F1、数值 MSE 与生成原文，排除反复输出空 items/全零数值仅满足 Schema 的情况。训练集记忆不等于验证集准确。
先观察多段连续训练及数据读取、验证、Checkpoint 的实际 T4 耗时/显存，不用单步数据外推全程。

满足后才人工审查并考虑 `exploratory_config.json`：完整 train=2338 / val=414，**1 epoch**，初始候选 batch 1、累积 8、lr 1e-4、checkpoint 每 50 更新、验证每 100 更新。
这些间隔是待实测调整的初始值；验证 loss 使用全 414 张，生成仍是固定子集，正式比较需另跑完整验证生成评价。
正式实验从原始模型新建 LoRA，不直接沿用 16 图过拟合权重；支持同配置完整断点恢复。不能用过拟合/smoke 断点混入全量实验。
Notebook 的 `RUN_FULL_TRAINING=False` 和 `SCHEMA_LEARNING_REVIEWED=False` 都默认关闭；不会自动启动默认 3 epochs。
