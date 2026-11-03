# 提示词复述：先定位，再决定是否训练

入口：[member_c_prompt_diagnostics.ipynb](member_c_prompt_diagnostics.ipynb)。使用成员 C 分支的新完整源码 SHA。
不更换模型、聊天模板、LoRA 参数或学习率，不自动训练，不恢复到 100 步，不修改旧实验。

## 已有事实与尚未证实的推断

以下为用户提供的真实 T4 实验结果，本地没有重新读取 Drive 或复跑 GPU：

| 指标 | Step 0 | Step 25 |
|---|---:|---:|
| Training loss | 未提供 | 0.99827 |
| Validation loss | 1.30534 | 1.06950 |
| Train JSON parse rate | 62.5% | 43.75% |
| Val JSON parse rate | 50% | 25% |
| Train / val Schema validity | 0 / 0 | 0 / 0 |
| Train prompt echo | 0/16 | 8/16 |
| Val prompt echo | 1/4 | 3/4 |

已提供的一条监督检查显示 13 个 512×512 图片块、832 个 image token、73 个监督 token、999 个屏蔽 token，matches_target=true。
这支持该条样本的 JSON 答案正确，但不足以证明其余 15 条、图像掩码和答案终止监督均正确。

代码检查中没有发现明确的提示词截取错误：训练/推理共用 user_messages，都是图片后接文本；输入由 Processor 展开图像 token，生成从带 Assistant 起点的提示继续。
当前 `generated[:, input_ids.shape[1]:]` 与 [官方模型快速示例](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) 的新增 token 截取一致。
[官方模型卡的推理说明](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct#how-to-get-started) 也采用先 apply_chat_template(add_generation_prompt=True)、再调用 Processor 处理图片的路线。
官方 [chat_template.json](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/main/chat_template.json) 的生成提示尾部为 Assistant:；运行报告以**固定 revision 实际加载的模板**为准，不把 main 的网页内容硬编码到程序里。
没有证据支持替换模板或修改 add_special_tokens/EOS，所以这些行为保持原样。

假如运行时返回序列的输入前缀完全一致，且“提示词片段”位于新 token 区域，就能排除把原输入切进答案这一解释：复述是模型生成行为。
复杂提示、有限样本下的条件生成/曝光偏差、尚未学会任务格式等都仍是候选原因；不能仅凭 loss 降低或复述增加断言哪一个成立。

## 新的代码证据与诊断文件

正常推理新增精确 token 前缀断言。若 generate 返回序列不是完整输入前缀加续写，会报错，不猜测切点、不靠文本替换修正。
仅显式 capture_trace 的诊断运行记录完整 token，不增加普通训练监控文件的体积：

- 实际提示词及 SHA256、未展开图像的模板文本及尾部。
- 展开图像后的 input_token_ids、full_output_token_ids、new_token_ids。
- 输入最后 160 个 token 的特殊 token 解码、完整返回序列和新增部分的特殊 token 解码。
- input_prefix_matches、图像形状、生成配置的 EOS/BOS/PAD/beam 等原始值。
- 新生成的特殊 token 的位置、EOS 是否结束以及生成是否到达 token 上限。

复述统计使用预先固定的启发式：输出中至少 8 个连续空白词与提示重叠，并占输出词数至少 50%。它不是教师指标，也不是根因证明；最终要查看原文和 token。
程序不会删除复述、重命名字段或以 target 替换答案。

## 下一次最小 GPU 运行

1. 审核并手动推送成员 C 分支，打开新 Notebook，填写本次完整 EXPECTED_CODE_COMMIT。
2. 选择 Colab NVIDIA GPU，逐格运行现有依赖保护、Drive ZIP 数据准备与固定版本模型缓存检查。
3. 可选填写旧 `EXISTING_STAGE25_DIR`（完整 schema_step25 目录，不是 checkpoint 子目录）。旧目录只读，输出都写入新的 `FoodLens/outputs/prompt_diagnostics_<ID>/probe/`。
4. 运行诊断单元，不运行原先的 100 步入口。模型只生成，不执行 backward/optimizer。

单独命令同样仅在 Colab 执行：

```bash
python -m task2_diet_record.prompt_diagnostics \
  --model-path <已缓存原始SmolVLM目录> --revision 7e3e67edbbed1bf9888184d9df282b700a323964 \
  --data-root /content/foodlens_data --output <新的Drive诊断目录> --max-new-tokens 512
```

可选追加 `--existing-stage25-dir <旧schema_step25目录>`。

### 全部 16 张训练图监督检查

`supervision_audit_16.json` 每张图片单独记录：

- target_text 与 target 是否一致，规范 JSON 是否等于真正被监督 token 的解码。
- 训练完整序列与推理提示的 token 前缀是否一致；实际输入/标签/attention mask/答案起点均保存。
- 用户提示、图片 token、padding 是否全部屏蔽；所有有效答案 token 是否受监督。
- 训练与推理的 pixel_values 是否完全相同，图片块形状和图像 token 数量。
- 是否超过最大长度、模板中答案后面的原始终止后缀、对应 token 是否完整受监督。
- 终止特殊 token 是否包含当前模型生成配置的 EOS。若不匹配，记录失败，不自动修改配置。

即使某张失败也继续检查剩余图片，汇总必须清楚区分 passed_samples 与 samples。
JSON 文本匹配但 EOS 标签被屏蔽的样本不会被判为全通过。报告检查的是答案边界，不把 JSON 与提示共享词汇误判为用户标签泄漏。
如果指定旧 Adapter，还使用其保存的 Processor 再检查 16 张，输出 `adapter_supervision_audit_16.json`。基础和 Adapter 模板不同时停止进一步 Adapter 生成，先解释差异，不能静默替换。

### A/B/C 零样本对照

| 组 | 提示 | 权重 |
|---|---|---|
| A | 原始完整 Schema 指令，PROMPT 未改 | 同一个未微调预训练模型 |
| B | 简短的 JSON 字段及数字要求 | 同上，无 Adapter |
| C | B + 人工规范 JSON 格式示例 | 同上，无 Adapter |

固定前 2 张 train 和前 2 张 val，共 4 张图，三个提示词各生成一次，共 12 次生成。统一 512 token 上限、seed 42、do_sample=False、use_cache=True，不改模板/裁图参数。
每组实际提示词和哈希保存在 manifest 和 A/B/C 报告，所有图片 ID 固定。按 train/val 分别记录解析率、Schema、F1、MSE、失败样本和复述启发式，不能把 4 图结果当成泛化成绩。
C 的示例是人工格式说明，不是这些图片的答案；报告额外标记完整照抄示例的情况。单纯复制示例得到合法 Schema 不算营养估计成功。
如果额外加载旧 25 步 Adapter，只运行原始 A，单独保存到 `trained_A_only/`；不把 B/C 与 A 的训练 checkpoint 混用。

## 何时才值得重新考虑到 100 步

先满足：16/16 监督检查通过，Assistant 起点和完整返回 token 前缀正常，EOS 与停止配置得到解释，图像处理一致，无截断导致的假象。
如果所有这些都正常，A 的提示复述确实出现在新增 token 内，且 B/C 在同图零样本中明显减少复述，才支持“提示复杂度相关”的下一步假设；它本身不足以证明短提示微调有效。
需要短提示训练时必须另建对应提示词签名与独立新实验，不能恢复 A checkpoint 后更换提示词再宣称 B/C 学会格式；当前入口根本没有训练功能。
只有同一训练协议下，原始输出开始出现可用 JSON/合法 Schema，复述没有继续恶化，且不靠照抄人工示例或伪修复，才重新审查是否继续原 A 实验到总计 100 步。
若仍为零合法输出且复述增加，优先解释报告，不继续堆 epoch/rank，也不自动开启完整训练。

本地只有无 GPU 单元测试和 Notebook 格式/语法校验，实际 16 样本报告、真实 token 轨迹与 A/B/C 结果需在 Colab 执行后产生。本次没有本地模型权重或原标注，不能伪造完整真实检查成绩。

本次本地共 110 项测试通过，包含原有 94 项；三份 Notebook 的格式与全部代码单元语法校验通过。新增测试使用人工 token 和模型替身，覆盖正确截取、真实新增区域中的复述、特殊 token 保留、EOS 未监督、用户/图像标签泄漏、16 条完整覆盖、提示词哈希/Checkpoint 隔离和旧目录只读保护。
