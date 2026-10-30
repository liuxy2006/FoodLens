# 成员 C：Colab GPU 工作流

## 当前阶段：16 图格式学习验证

用户已反馈 `8b06fd88…` 在 T4 完成两步真实集成测试，但 Schema 合法率仍为 0。
下一步使用独立 [Schema 实验 Notebook](member_c_schema_overfit.ipynb) 和 [操作与诊断说明](SCHEMA_EXPERIMENT.md)，先 25 步、人工审查后最多 100 步。
不要重跑或覆盖已有 smoke 目录。该实验的训练子集评价只表示记忆能力；目前尚无这轮小实验的 GPU 结果。

入口为 [member_c_train.ipynb](member_c_train.ipynb)。本地只开发和运行不需要 GPU 的测试；所有后续视觉模型推理与训练在 Colab NVIDIA GPU 进行。
Notebook 调用仓库模块，不复制训练循环。下面保留初期集成测试的运行说明；两步训练链路已由用户实际运行验证，不代表 Schema 学习通过或正式模型精度。

## 本地已执行的检查

2026-10-29 确认分支为 `feat/member-c-diet-record`，main 仍为 `0471f1c`。
此前创建了项目 `.venv`，初始化了 pip，并开始下载官方 XPU PyTorch；用户改为 Colab 路线后已中止，检查时 torch/transformers/peft 均未安装。
保留环境文件，不卸载或删除。没有在本地下载模型权重，也没有本地 CPU/GPU 模型训练。

只查询了官方 SmolVLM 文件元数据：revision `7e3e67edbbed1bf9888184d9df282b700a323964`，必要文件合计 516,614,854 字节，safetensors 文件 513,028,808 字节。
该信息不是模型加载或训练测试成绩。

本地原有 30 项和新增 13 项测试已实际通过（共 43 项）；Notebook 代码单元语法与 nbformat 格式校验通过。
新增测试覆盖评价分母、匹配原料重量、断点校验和配置匹配，以及通过替身验证的 Processor/LoRA 调用路径，不代表真实 GPU 或权重加载已通过。

2026-10-30 小范围修订后共 64 项本地测试通过，新增 ZIP 层级识别/缓存复用/文件完整性、依赖安装防护和完整 commit SHA 检查；Notebook 格式及所有代码单元语法检查通过。本次没有在本地安装训练依赖、下载权重或训练模型。

## 使用步骤

1. 审核本地提交后，由你决定何时推送 `feat/member-c-diet-record`；本阶段不会自动推送。远程必须包含第二阶段代码，Notebook 检查不到所需文件就停止。
2. 将 Notebook 上传到 Colab，选择「运行时 → 更改运行时类型 → GPU」（例如 T4）。Notebook 不允许 CPU 训练回退。
3. 将两个 ZIP 上传到 Google Drive（不要在 Drive 解压）：

   ```text
   MyDrive/FoodLens/dataset/
     annotations.zip
     images.zip
   ```

4. 推送后在本地执行 `git rev-parse HEAD`，把最终完整 40 位 SHA 填入 Notebook 的 `EXPECTED_CODE_COMMIT`。不能使用旧的 `6a8ac50`、短 SHA 或空值。Notebook 只克隆成员 C 分支，并严格比较实际 HEAD；不一致就停止，不自动切换 main 或覆盖已有代码目录。
5. 逐格运行 Notebook：克隆指定分支、保护 CUDA 的依赖安装、挂载 Drive、复制 ZIP 并准备本地数据、核实并下载固定 revision、未微调推理、1 步 LoRA、恢复到总计 2 步、验证集小样本评价。
6. 检查 Drive 下 `FoodLens/outputs/member_c_<运行ID>/` 的真实报告。每次使用新目录，旧实验不被覆盖。

`/content` 是临时运行目录，重置运行时后会丢失；原始 ZIP、模型缓存、断点和结果保存在 Drive。
不要把 token 写入 Notebook；此公共模型不需要额外提供私钥或凭据。

## 本地准备 ZIP 与 Colab 解压

已有两个 ZIP 时直接上传即可。若需要从当前项目数据生成，可在项目根目录运行下面的 Python 代码（支持 ZIP64，输出目录被 Git 忽略；`mode='x'` 拒绝覆盖已有 ZIP）：

```python
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

output = Path('outputs/colab_upload')
output.mkdir(parents=True, exist_ok=True)
for name, source in [('annotations', Path('data/raw/annotations')),
                     ('images', Path('data/images'))]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    with ZipFile(output / f'{name}.zip', mode='x', compression=ZIP_DEFLATED,
                 allowZip64=True) as archive:
        for file in sorted(source.rglob('*')):
            if file.is_file():
                archive.write(file, (Path(source.name) / file.relative_to(source)).as_posix())
```

`annotations.zip` 可包含顶层 `annotations/`、更深的包装目录，或直接包含三份 JSONL，但只能有一个同时包含 train/val/test.jsonl 的目录。
`images.zip` 可包含顶层 `images/`，也可直接包含 train/val/test，允许更深包装目录；模块识别三个 split 的唯一共同父目录，避免 `images/images/`。

Notebook 调用 `prepare_colab_data.py`：检查源 ZIP → 复制到 `/content/foodlens_archives/` 并核对 SHA256 → 在本地安全解压、规范目标路径 → 验证数量及全部图片引用 → 写入完成标记。结果固定为：

```text
/content/foodlens_data/
  raw/annotations/train.jsonl   # 2338 条
  raw/annotations/val.jsonl     # 414 条
  raw/annotations/test.jsonl    # 506 条
  images/train/...
  images/val/...
  images/test/...
```

`DATA_ROOT=/content/foodlens_data`。三份标注原始字节、图片内容均不改写；test 只核对完整性，不进入训练或选模。
复跑时校验源 ZIP、完成标记、已提取文件哈希和全部引用，正确则复用，不重复复制/解压。文件缺失、数量不符、目录歧义、数据变更均明确报错。
已有不完整目录或缓存不会被清理或覆盖；请人工检查，或给 `DATA_ROOT` / `ARCHIVE_CACHE` 选择新的 `/content/...` 目录重试。
数据准备结果写入当前实验的 `data_preparation.json`。

## 环境与模型

保留 Colab 原有 CUDA PyTorch，固定 `transformers==4.51.3`、`peft==0.15.2`、`accelerate==1.6.0`、`huggingface-hub==0.30.2`、`tokenizers==0.21.4`。
已核对官方发布依赖元数据与 Transformers 4.51.3 的 Idefics3 API；该 API 包含本项目使用的 `AutoProcessor` / `AutoModelForVision2Seq` 支持。当前固定版本不含 Windows/XPU GPU 包，但 PEFT/Accelerate 有 torch 传递依赖，不能只凭 requirements 没写 torch 就认为不会替换运行时。
安装统一调用 `colab_dependencies.py`：先在新进程读取并打印 `torch.__version__`、`torch.version.cuda`、`torch.cuda.is_available()`，用已安装版本生成精确约束，再审查 pip 的 dry-run 安装清单。
若计划安装/替换 torch、torchvision、torchaudio、triton、NVIDIA/Intel 运行时包或 Windows/XPU wheel，安装前立即停止。
安装后再次在新进程检查上述三项，版本/CUDA 构建改变或 CUDA 失效会明确报错并停止；不自动修复或安装另一套 PyTorch。兼容冲突会让解析失败，而不是升级 GPU 环境。
安装报告先保存在 `/content/foodlens_install_<ID>/`，成功挂载 Drive 后复制到实验目录的 `dependency_install/`，包含安装前后快照、约束、计划和实际安装记录。
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

真正通过 GPU 集成测试，应同时看到：CUDA 安装前后可用且版本未变、推理报告有真实生成文本、训练 loss 有限且梯度非零、更新参数张量数量大于 0、恢复日志从第 1 步接到第 2 步、`identical_text=true`、验证评价成功落盘。仅有 Notebook 格式/单元测试通过，不能代表 GPU 训练已通过；JSON/Schema 失败率则应按原样报告。

JSON 不合法不等于训练框架失败，未微调模型可能不能按 Schema 输出；训练数步也不保证营养准确。
反向/优化器失败、非有限值、无参数更新、重载文本不一致属于必须排查的问题，不能虚报通过。
有效子集与失败补值的统计分母不同；匹配原料重量误差有选择偏差，只作附加诊断。

Notebook 中 `RUN_FULL_TRAINING=False`，不会自动训练全部 2338 张图片。
确认两步集成测试通过后再按 GPU 显存和每步耗时决定正式参数；初始候选仍是 batch 1、累积 8、rank 8、lr 1e-4，先完成一个 epoch 检查验证集表现。
没有 Colab 实测之前，不能判断完整训练耗时或保证当前配置的峰值显存。

官方参考：[Colab FAQ](https://research.google.com/colaboratory/faq.html)、[PEFT checkpoint](https://huggingface.co/docs/peft/developer_guides/checkpoint)、[SmolVLM 模型](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)。
依赖核对：[Transformers 4.51.3](https://pypi.org/pypi/transformers/4.51.3/json)、[PEFT 0.15.2](https://pypi.org/pypi/peft/0.15.2/json)、[Accelerate 1.6.0](https://pypi.org/pypi/accelerate/1.6.0/json)、[Idefics3 4.51.3 文档](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/idefics3)。
