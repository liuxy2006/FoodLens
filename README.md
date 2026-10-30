# FoodLens

神经网络与深度学习课程设计。

## 项目任务

1. 食物图像描述生成
2. 食物图像到结构化膳食记录
3. LLM-as-Judge 评价
4. 单图像营养估计

## 项目结构

- `data/`：数据集说明和数据处理
- `task1_caption/`：食物图像描述生成
- `task2_diet_record/`：结构化膳食记录生成
- `task3_llm_judge/`：LLM-as-Judge
- `task4_nutrition_estimation/`：单图像营养估计
- `evaluation/`：BLEU、CIDEr、F1、MSE 等指标
- `app_or_demo/`：统一演示入口
- `configs/`：模型和实验配置
- `docs/`：开题、中期和结题资料
- `tests/`：测试代码

## 小组分工

- 成员 A：数据处理、任务一模型 A
- 成员 B：任务一模型 B、LLM-as-Judge
- 成员 C：任务二、单图像营养估计

## 成员 C 开发入口

数据审计、结构化 JSON、F1/MSE、营养辅助基线及 SmolVLM LoRA 框架已建立。
运行方式、指标约定、实际数据检查结果和训练前置条件见 [成员 C 文档](task2_diet_record/README.md)。
当前没有已训练的成员 C 模型，接口测试结果不能作为模型实验成绩。

成员 C 的后续模型训练统一在 Google Colab NVIDIA GPU 进行，入口见 [Colab Notebook](colab/member_c_train.ipynb) 和 [运行说明](colab/README.md)。本地仅运行无需 GPU 的开发测试。
