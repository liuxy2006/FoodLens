# 成员 C：第一阶段

> **2026-10-29 第二阶段更新：训练已迁移到 Google Colab NVIDIA GPU。**
> 优先使用 [Colab 运行说明](../colab/README.md) 和 `colab/member_c_train.ipynb`。
> 下方本地 CPU/XPU 训练准备段是第一阶段历史方案，不再作为当前执行步骤；本地仅运行无需 GPU 的测试。
> 当前新增了可恢复断点与保存重载检查，但真实 GPU 结果须在 Colab 执行后报告。

所有命令在 `D:\nnproject` 执行。原始数据只读，默认数据根目录为 `data`。
当前实现范围是数据检查、评价、营养基线、可配置的 LoRA 训练框架和真实推理接口。
尚未安装训练依赖、下载权重或训练视觉语言模型，不能报告模型准确率。

## 模块和数据流程

- `data.py`：逐行读取标注、限制图片路径位于数据目录、加载 RGB 图片、核对监督文本。
- `audit.py`：分别执行数据质量审计与 Schema 审计，检查全部图片和跨集合泄漏。
- `schema.py`：兼容 `nutrition5k-student-v1` 的输出结构。仓库没有教师独立 Schema，因此暂以标注 target 定义；后续拿到正式版本再核对。
- `interfaces.py`：`DietModel.predict(image_path) -> raw_text`；A/B 的 Caption 模型通过 `CaptionDietAdapter` 接入，提取器由调用方明确提供。没有默认假模型。
- `vlm.py`：真实图片 Processor 输入、模型加载、仅 assistant 答案的损失掩码。
- `train.py`、`train_config.json`：语言注意力 q/v 投影 LoRA、冻结其余参数、梯度累积、梯度裁剪、验证、最佳和最后 adapter 保存、实验配置及版本记录。
- `predict.py`：批量离线推理，失败仍保存该样本与错误；不把标注 target 或 dish_id 传给模型。
- `../evaluation/diet_metrics.py`：统一评价和按 dish_id 对齐的命令入口。
- `../task4_nutrition_estimation/baseline.py`：非负最小二乘营养基线，以及 direct/ingredients 两种图片营养估计接口。
- `../app_or_demo/diet_cli.py`：单图真实推理演示入口，后续可包装上传页面。

训练流程：RGB 图片 + 固定提示 → Processor → 预训练视觉编码器与语言模型 → assistant JSON 的因果语言建模损失。
批处理分别编码完整对话与提示，确认提示 token 严格是前缀，再把提示及 padding 标签设成 `-100`。
图片 token 位于提示部分，也不作为答案监督；不会截断图像 token 或半个 JSON，超过长度会明确报错。
验证损失采用 teacher forcing（用于模型选择），额外生成验证和正式推理仅接收图片与固定提示。

## 已实际执行的数据检查（2026-10-28）

| 项目 | train | val | test |
|---|---:|---:|---:|
| 标注与成功加载图片 | 2338 | 414 | 506 |
| Schema 合法且 target_text 一致 | 2338 | 414 | 506 |
| 原料种类 | 151 | 100 | 118 |
| 原料重量和偏差超过 0.1 g | 1214 | 296 | 260 |

共 3258 张图片，没有未引用图片、缺失或损坏图片，也没有重复 dish_id。
跨集合 dish_id、图片路径、文件 SHA256、解码 RGB 像素 SHA256 重叠均为零。
这不代表排除了近似图片、同一菜品不同视角等语义泄漏。
原料和总重量偏差属于数据质量现象，不会导致 Schema 失败，也不会自动修正标签。
测试集只做用户要求的静态完整性检查，不用于拟合、模型选择或提示词调整。

完整结果保存在 `outputs/data_audit.json`，包括每种原料按菜品统计的出现次数、原料数量分布、五项营养数值及原料重量的 min/max/mean/std/分位数、异常清单和标注 SHA256。
报告与拟合参数不进入 Git；可随时复跑：

```powershell
python -m task2_diet_record.audit --data-root data --output outputs/data_audit.json
python -m task2_diet_record.environment
python -m unittest discover -s tests -v
python -m task4_nutrition_estimation.baseline --train data/raw/annotations/train.jsonl --output outputs/nutrition_baseline.json
```

预期检查数量与上表一致；测试输出 `OK`；基线输出 `151 ingredients`。
若报找不到图片，先检查 `data/images/train/...`，不要把根目录误设成项目目录或 `data/raw`。
图片损坏、JSON 结构不符会记录或明确报错，不修改原文件。

## 指标约定

名称采用 Unicode NFKC、casefold、合并连续空白与首尾去空白。按每张图的原料名称集合匹配，重复名称只计一次。
不自行做单复数、同义词合并，避免未经验证的语义映射。
micro 先累加整个数据集 TP/FP/FN，再算 Precision、Recall、F1；macro 是各图片指标的算术平均。
零分母指标定义为 0（含双方空原料集合）。

MSE 对五个字段独立计算，单位分别为 g²、kcal²、g²、g²、g²，不把不同量纲直接平均。
JSON 解析率、Schema 合法率的分母都是全部应预测样本。
失败、缺失或 Schema 非法的预测统一视为空原料集合和全零营养，进入明确标注为失败补值的 `mse_all_zero_fallback`，它不能无条件称为标准营养预测 MSE。
这是一种明确、可复现的失败处理规则，不声称零值是科学估计；必须同时报告失败率。
`valid_only` 分别报告合法预测的 micro/macro F1 和五项 MSE，分母为合法预测数量；必须同时报告失败率。
`all_samples_with_failure_policy` 使用全部样本，明确写出空原料和全零营养的失败处理规则。
保留第一阶段字段作为兼容别名；没有合法预测时，有效子集指标为 null。
按匹配原料计算的重量 MAE/MSE 仅为附加诊断，会遗漏未匹配原料，不能替代教师规定的指标。
严格拒绝重复 JSON key、NaN/Infinity、负数、额外字段和非 JSON 解释文本；允许完整 JSON 的单个 Markdown 围栏。

预测文件每行 `{"dish_id":"...","prediction":"模型原始文本"}`；推理失败 prediction 为 null 并记录 error。
评价按标注对齐，缺少的预测算失败，多余或重复 dish_id 报错。

```powershell
python -m evaluation.diet_metrics --annotations data/raw/annotations/val.jsonl --predictions outputs/val_predictions.jsonl --output outputs/val_metrics.json
```

接口测试使用明确的人工构造样例，完美匹配/失败分支的指标只用于验证公式，不是模型成绩。
目前未重复开发任务一的 BLEU/CIDEr；保留 A/B 接口，待其模型和评价模块到位再统一比较。

## 营养辅助基线

只用 train 中真实原料克数构建设计矩阵 X，用 NNLS 拟合四项非负营养系数 B，使 `X B ≈ Y`。
总重量采用**预测**原料重量之和；四项营养为预测重量与系数乘积。
未见原料采用训练集总体每克营养均值，空原料返回零。
系数是训练数据的统计关系，受可见原料不完整、标注重量偏差和共线性影响，不是权威营养成分表。
`fit()` 明确拒绝 val/test。`NutritionEstimator` 的 direct 和 ingredients 都只从图片开始调用模型；后者只消费模型预测的 items。
实际执行了训练集拟合并保存参数，但没有拿 val/test 真值原料评测端到端模型。

## 硬件与模型选择

本机实测 Python 3.13.11，Intel Arc Graphics，系统内存约 31.48 GiB，D 盘可用约 555 GiB。
未安装 PyTorch/Transformers/PEFT/Accelerate，`nvidia-smi` 不可用；CUDA/XPU 可用性和专用显存尚未由框架确认，不能认为已有 NVIDIA 训练环境。
核心数据检查依赖已经可用，实际版本写入 `outputs/environment.json`。

| 候选 | 方法 | 当前取舍 |
|---|---|---|
| [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | Transformers 图片输入，语言投影 LoRA | 推荐先跑通，小规模参数更适合资源有限环境；营养数值精度须实测 |
| [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | Transformers，LoRA；兼容 CUDA 设备可另评估 QLoRA | 模型更大、资源开销更高；作为有合适设备时的后续候选 |

以上推荐是基于模型规模和官方接口的工程判断，不是本项目精度比较。
官方参考：[SmolVLM 文档](https://huggingface.co/docs/transformers/model_doc/smolvlm)。
当前训练框架针对 SmolVLM/Idefics3 的 text_model 注意力路径，**没有声称已兼容 Qwen2-VL 训练**。
本阶段不依赖 bitsandbytes，不假设 Intel Arc 支持 CUDA QLoRA。

## 训练准备与命令（尚未执行）

建议另建 Python 3.11 环境。需要你确认后才安装：核心 `requirements.txt`，按设备选择的 PyTorch，以及 `task2_diet_record/requirements-train.txt`。
CPU 路线可考虑如下安装命令；CUDA/XPU 环境须先确认型号并选择官方对应 wheel，不能照搬：

```powershell
python -m pip install -r requirements.txt
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r task2_diet_record/requirements-train.txt
```

模型下载同样需要确认。仓库名在配置里，下载时应锁定模型仓库的完整 commit SHA，保留模型缓存。
默认所有模型加载均为 `local_files_only=True`。仅训练命令显式加 `--allow-download --revision <完整40位SHA>` 才允许下载。
模型存放 `models/SmolVLM-256M-Instruct` 等 Git 忽略目录。没有权重时推理会明确失败，不输出伪造答案。

先执行 1 个优化器步（默认累积 8 个 batch）和少量验证：

```powershell
python -m task2_diet_record.train --model-path models/SmolVLM-256M-Instruct --device cpu --max-steps 1 --val-limit 2 --generate-val 2 --output outputs/smolvlm_smoke_01
```

CPU 仅用于确认流程，速度须现场测量。XPU 路线必须先确认 PyTorch 可识别 Intel 设备和全部模型算子。
有可用 CUDA/XPU 后可改对应 `--device`；默认 float32/eager，未做混合精度或 4-bit 显存承诺。
预期输出可训练参数量、有限的 train/val loss、生成合法率/F1/MSE、best/last adapters 和日志。
首次真实集成必须检查：图片 token 数量、答案掩码、LoRA 参数有梯度、损失有限、同一提示换不同图片确实送入不同 pixel_values。
这些真实模型检查目前因缺少训练依赖及权重未执行；张量替身测试只验证掩码算法。

正式训练（先确认设备与耗时）：

```powershell
python -m task2_diet_record.train --model-path models/SmolVLM-256M-Instruct --device cpu --generate-val 32 --output outputs/smolvlm_run_01
python -m task2_diet_record.predict --model-path models/SmolVLM-256M-Instruct --adapter outputs/smolvlm_run_01/best --annotations data/raw/annotations/val.jsonl --device cpu --output outputs/val_predictions.jsonl
python -m evaluation.diet_metrics --annotations data/raw/annotations/val.jsonl --predictions outputs/val_predictions.jsonl --output outputs/val_metrics.json
```

配置默认 3 epochs、batch 1、梯度累积 8、seed 42、lr 1e-4、LoRA rank 8。
只读 train/val，按 val loss 保存 best；生成验证可选，用于监控实际输出。
日志保存配置、库版本、模型解析 revision、标注 SHA256、样本数、训练/验证损失和可选生成指标。
已有实验目录会被拒绝覆盖；best/last 在同一实验内部按训练进度更新。
第二阶段已新增 `--resume` 完整断点恢复，详见 Colab 文档；断点包含 adapter、Processor、优化器、随机状态和样本游标。
CPU 浮点和训练随机性并不保证跨设备完全一致，务必保存运行日志。
锁定方案之后再把推理/评价命令中的 val 改成 test 做最终测试，禁止用 test 调参。

单图演示：

```powershell
python -m app_or_demo.diet_cli --image data/images/train/dish_1559149615.png --model-path models/SmolVLM-256M-Instruct --adapter outputs/smolvlm_run_01/best
python -m app_or_demo.diet_cli --image data/images/train/dish_1559149615.png --model-path models/SmolVLM-256M-Instruct --adapter outputs/smolvlm_run_01/best --method ingredients --baseline outputs/nutrition_baseline.json
```

如果显存不足，先减少图像分辨率/图像切片和 batch（需相应固定实验配置），不要删除缓存或偷偷截断监督答案。
如果生成失败率高，先保留原文、检查 EOS、生成长度和训练掩码；不得手工改成真值后再计分。
