# 在 Kaggle 运行 FoodLens 提示词诊断

入口：[member_c_prompt_diagnostics_kaggle.ipynb](member_c_prompt_diagnostics_kaggle.ipynb)。
实验和原 [Colab 方案](../colab/PROMPT_DIAGNOSTICS.md) 相同：16 张训练图的监督审计，以及固定前 2 张 train 和前 2 张 val 的 A/B/C 共 12 次零样本生成。不训练 LoRA，不启动 100 步或全量训练。

## 逐格运行

1. 审核并自行推送 `feat/member-c-diet-record`。用 `git rev-parse HEAD` 取得推送的完整 40 位 SHA。将 Kaggle Notebook 文件导入自己的**私有** Kaggle Notebook，在第一格设置 `EXPECTED_CODE_COMMIT`。它只使用成员 C 分支；若该分支的 HEAD 和填写的 SHA 不同，立即停止。若本会话已有其他版本的克隆，开启新会话再运行。
2. 创建或更新一个私有 Kaggle Dataset，将原始 `annotations.zip` 和 `images.zip` 放在同一目录。保持 ZIP 文件原样，通过 **Add Input** 将 Dataset 挂载到 Notebook。默认在 `/kaggle/input` 内查找唯一一对 ZIP；如果挂载了多个版本，在 `DATASET_DIR` 填写目标目录的完整挂载路径，或相对 `/kaggle/input` 的路径。无需改用户名、重命名 ZIP 或预解压。
3. 在会话设置中选择 **GPU T4**（可用时界面可能显示 **GPU T4 x2**）并打开 **Internet**。需要访问 GitHub、PyPI 和 Hugging Face。诊断只使用第一个可见 CUDA 设备；不需要双卡并行。参见 [Kaggle 官方 Notebook 指南](https://www.kaggle.com/docs/notebooks)。
4. 运行代码校验和输入路径单元，再运行 CUDA 与依赖预检。它在安装前后用新进程实测 CUDA float32 矩阵乘法。pip 先输出 dry-run 清单；如需替换预装 CUDA PyTorch 组件，立即停止。沿用项目的 Transformers 4.51.3、PEFT 0.15.2、Accelerate 1.6.0，并测试这些包和 SmolVLM API 能否导入。若失败，查看错误和 `cuda_before.json`、`cuda_after.json` 或 `dependency_install/` 内已生成的报告；不要手工重装 torch 来绕过检查。
5. 运行数据和模型准备单元。ZIP 只读挂载，复制和解压到 `/kaggle/temp/foodlens_prompt_diagnostics`，核对原有 2338 train / 414 val / 506 test 数量及图片路径，test 仅做完整性检查。随后从官方仓库下载 `HuggingFaceTB/SmolVLM-256M-Instruct` 的固定 revision `7e3e67edbbed1bf9888184d9df282b700a323964`，模型和 Processor 来自同一版本。下载通过大小和 LFS 校验；同一会话会复用缓存，新会话可能需要重新下载。
6. 运行诊断和最后的报告单元。默认 `EXISTING_STAGE25_DIR = ""`，无需旧权重。检查 16 条监督报告及三组各 4 条原始预测，再下载报告。

### 可选：附带旧 Colab Adapter

把完整的 `schema_step25` 目录（含 `run.json`、`steps.jsonl`、`monitor/step-000025.json`、`last_checkpoint.json` 和 `checkpoint-000025/`）作为另一个私有 Dataset 挂载。将 `EXISTING_STAGE25_DIR` 指向 Kaggle 实际挂载的实验目录。
Notebook 在临时副本中把旧 Colab 的绝对断点指针转换为相对路径，原文件和旧报告保持不变。原诊断模块仍核验 checkpoint 文件完整性、配置、数据、模型版本和提示词签名；通过后，另外对同样 4 张图运行原提示词 A。Adapter 结果在 `trained_A_only/`，基础 A/B/C 始终使用未微调模型。

## 下载和核对报告

每次运行创建独立的 `/kaggle/working/foodlens_prompt_diagnostics/<运行 ID>/`。最后一格在同一父目录打包 `<运行 ID>.zip`，可以在 Kaggle 的 **Output** 面板下载。若使用 **Save Version → Save & Run All**，Kaggle 会在新会话重新运行全部单元，生成新的报告，消耗相应的 GPU 时间。
压缩包包括版本、CUDA/依赖、模型和数据准备记录，以及：

- `probe/supervision_audit_16.json`：检查是否覆盖全部 16 条，逐条核对图片、Label Mask、Assistant 起点、截断和 EOS。
- `probe/zero_shot/manifest.json` 与 `A.json`、`B.json`、`C.json`：确认固定图片、提示词和哈希，核对 12 条原始预测的输入前缀、新增 token、终止信息、提示词复述和失败记录。要判断真实生成是否成功，也要检查逐条 `error`，不能只数报告记录。
- `probe/zero_shot/summary.json`：各组的解析、Schema、原料与营养指标。四张图不足以推断泛化；C 组照抄示例不能算预测准确。

报告可能含课程数据的标注内容，请保持 Notebook 和报告私有。模型权重、数据和缓存都不在报告压缩包中。本迁移仅做本地无 GPU 校验，真实 Kaggle T4 结果须由运行生成。
