# 成员 C：Colab GPU 工作流

入口为 [member_c_train.ipynb](member_c_train.ipynb)。本地只开发和运行不需要 GPU 的测试；所有后续视觉模型推理与训练在 Colab NVIDIA GPU 进行。
Notebook 调用仓库模块，不复制训练循环。当前没有实际 Colab GPU 运行结果，不能声称 LoRA 已训练成功。

## 本地已执行的检查

2026-10-29 确认分支为 `feat/member-c-diet-record`，main 仍为 `0471f1c`。
此前创建了项目 `.venv`，初始化了 pip，并开始下载官方 XPU PyTorch；用户改为 Colab 路线后已中止，检查时 torch/transformers/peft 均未安装。
保留环境文件，不卸载或删除。没有在本地下载模型权重，也没有本地 CPU/GPU 模型训练。

只查询了官方 SmolVLM 文件元数据：revision `7e3e67edbbed1bf9888184d9df282b700a323964`，必要文件合计 516,614,854 字节，safetensors 文件 513,028,808 字节。
该信息不是模型加载或训练测试成绩。

本地原有 30 项和新增 13 项测试已实际通过（共 43 项）；Notebook 代码单元语法与 nbformat 格式校验通过。
新增测试覆盖评价分母、匹配原料重量、断点校验和配置匹配，以及通过替身验证的 Processor/LoRA 调用路径，不代表真实 GPU 或权重加载已通过。

## 使用步骤

1. 审核本地提交后，由你决定何时推送 `feat/member-c-diet-record`；本阶段不会自动推送。远程必须包含第二阶段代码，Notebook 检查不到所需文件就停止。
2. 将 Notebook 上传到 Colab，选择 NVIDIA GPU 运行时。Notebook 不允许 CPU 训练回退。
3. 将数据上传并解压到 Google Drive：

   ```text
   MyDrive/FoodLens/data/
     raw/annotations/train.jsonl
     raw/annotations/val.jsonl
     raw/annotations/test.jsonl
     images/train/...
     images/val/...
     images/test/...
   ```

4. 逐格运行 Notebook：克隆指定分支、安装 Python 依赖、挂载 Drive、检查数据、核实并下载固定 revision、未微调推理、1 步 LoRA、恢复到总计 2 步、验证集小样本评价。
5. 检查 Drive 下 `FoodLens/outputs/member_c_<运行ID>/` 的真实报告。每次使用新目录，旧实验不被覆盖。

`/content` 是临时运行目录；模型缓存、断点和结果保存在 Drive。可以把数据复制到 `/content` 加快读图，但 Notebook 默认直接读取 Drive，不执行大规模复制或解压覆盖。
不要把 token 写入 Notebook；此公共模型不需要额外提供私钥或凭据。

## 环境与模型

保留 Colab 原有 CUDA PyTorch，固定 `transformers==4.51.3`、`peft==0.15.2`、`accelerate==1.6.0`、`huggingface-hub==0.30.2`、`tokenizers==0.21.4`。
实际可用版本要以 Colab 的导入检查、`pip_freeze.txt`、`device_probe.json` 和 `nvidia_smi.txt` 为准；这些组合尚未在当前会话的远程 GPU 验证。
若依赖变更要求重启，请重启会话，再从头执行。若预装 torchvision 和 torch 不匹配，应先记录错误，使用匹配的官方软件包，不能盲目忽略导入异常。

模型仅为 `HuggingFaceTB/SmolVLM-256M-Instruct`，固定上述 revision。
`prepare_model.py` 先列出必要文件，显式 `--download` 才下载，并核对文件大小及 LFS SHA256。
模型与 Processor 使用同一份版本；Adapter 重载时使用断点保存的 Processor。
默认 float32/eager、batch 1、梯度检查点，不引入 QLoRA 或第三方加速库。

## 训练和恢复行为

训练仍由 `task2_diet_record.train` 执行，仅读取 train/val：

```bash
python -m task2_diet_record.train --config /path/to/smoke_config.json \
  --model-path /path/to/model --revision 7e3e67edbbed1bf9888184d9df282b700a323964 \
  --data-root /path/to/data --device cuda --train-limit 8 --val-limit 2 \
  --max-steps 1 --checkpoint-every 1 --verify-update --verify-reload \
  --output /path/to/new_run
```

Notebook 自动生成相同 smoke 配置：1 epoch、batch 1、累积 1，8 张 train 样本。
`--max-steps` 是**包含已恢复步骤的总优化器步数上限**，不是“再训练几步”。恢复到总计 2 步：

```bash
python -m task2_diet_record.train <相同数据/模型/配置参数> \
  --resume /path/to/old_run/checkpoint-000001 --max-steps 2 --output /path/to/new_resume_run
```

完整断点目录包括：

- Adapter safetensors 与配置、Processor/tokenizer。
- `training_state.pt`：优化器和 Python/NumPy/PyTorch/CUDA 随机状态。
- `metadata.json`：下一轮/轮内样本位置、已完成步数、最佳验证损失、配置及数据/模型签名。
- `READY.json`：最后写入的文件 SHA256 清单；不完整或被改动的断点拒绝恢复。

每轮样本顺序由 seed+epoch 的独立生成器确定，恢复时跳过已完成样本；只在优化器更新后保存，没有未保存的半组梯度。
恢复前检查模型 revision/config、标注哈希、训练样本数量、验证子集大小和全部训练配置，不允许用 smoke 断点偷偷恢复全数据训练。
输出目录必须新建。`last_checkpoint.json` 指向本次最新完整断点，`best_adapter.json` 指向最佳验证 adapter（可能位于此前的 Drive 实验目录）。请保留关联目录。
Drive 断连或进程中断后，选择最后一个能通过 `inspect_checkpoint()` 的 READY 断点；本工具不会清理不完整目录。
相同硬件/版本下应尽量复现，跨 GPU/库版本不承诺逐位一致。

## 实际 GPU 运行后检查什么

- `pretrained_inference.json`：真实图片 ID、原始输出、JSON/Schema 结果、时间和 GPU 显存峰值。
- `run.json`：源码 commit、模型版本、LoRA 模块、总参数/可训练/冻结参数数量、依赖和数据签名。
- `steps.jsonl`：真实 loss、非零且有限梯度、实际更新的参数张量数量、单步时间和 GPU 内存峰值。
- `reload_check_*.json`：保存前模型与从磁盘重载的模型，对相同验证图片的贪心生成文本是否一致。
- `metrics.jsonl`：验证损失、可选生成评价、资源使用情况。
- `val_metrics.json`：解析率、合法率、有效预测指标、全样本失败补值指标和附加重量误差。

JSON 不合法不等于训练框架失败，未微调模型可能不能按 Schema 输出；训练数步也不保证营养准确。
反向/优化器失败、非有限值、无参数更新、重载文本不一致属于必须排查的问题，不能虚报通过。
有效子集与失败补值的统计分母不同；匹配原料重量误差有选择偏差，只作附加诊断。

Notebook 中 `RUN_FULL_TRAINING=False`，不会自动训练全部 2338 张图片。
确认两步集成测试通过后再按 GPU 显存和每步耗时决定正式参数；初始候选仍是 batch 1、累积 8、rank 8、lr 1e-4，先完成一个 epoch 检查验证集表现。
没有 Colab 实测之前，不能判断完整训练耗时或保证当前配置的峰值显存。

官方参考：[Colab FAQ](https://research.google.com/colaboratory/faq.html)、[PEFT checkpoint](https://huggingface.co/docs/peft/developer_guides/checkpoint)、[SmolVLM 模型](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)。
