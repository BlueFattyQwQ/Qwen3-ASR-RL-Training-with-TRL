# 使用 TRL (GRPO) 对 Qwen3-ASR 进行强化学习训练

本仓库提供一个基于 Hugging Face **TRL** 的 **Qwen3-ASR 强化学习训练脚本**。

主脚本：
- `qwen3_asr_gspo.py`

## 项目简介

该项目使用 TRL 的 `GRPOTrainer`，通过自定义奖励函数优化 ASR 转写结果。

核心思路：
- 以音频路径作为 prompt。
- 使用 Qwen3-ASR 生成转写文本。
- 基于 CER 计算奖励（`reward = 1 - CER`）。
- 将自定义 rollout 结果（`prompt_ids`、`completion_ids`、`logprobs`）回传给 GRPO。

## 脚本如何接入 TRL

这份脚本通过 3 个关键部分完成 TRL 集成：

1. `asr_reward_func(...)`
- 接收 TRL 拉平后的生成文本和参考文本。
- 如有 `<asr_text>` 前缀会先去掉。
- 通过 `simpleCER.getCER` 计算 CER。
- 返回与 completion 数量一致的标量奖励列表。

2. `my_rollout_func(prompts, trainer)`
- 实现了 TRL 需要的自定义 rollout 接口（适配音频输入）。
- 遍历每条音频路径，调用 `generate_single_rollout(...)`。
- 最终返回批量的 `prompt_ids`、`completion_ids`、`logprobs`。

3. `GRPOTrainer(...)`
- 将模型、奖励函数、数据集、processor 和 rollout 函数组装到一起。
- 通过 `GRPOConfig` 控制训练和采样参数（`num_generations`、`temperature`、`top_p`、`beta` 等）。

## 为什么需要补丁（Patch）

脚本里有两个运行时补丁，用于提升与 Trainer 的兼容性：

- **Patch 1**：将 Qwen3-ASR 类注册到 `transformers` 自动模型注册表中。
- **Patch 2**：在外层包装模型上补充/覆盖 `get_input_embeddings`、`set_input_embeddings` 和 `forward`，并委托到 `model.thinker`。

这两个补丁可以避免 TRL 期望标准 Causal LM 接口时出现的常见报错。

## 数据格式

训练数据通过 `datasets.load_dataset("json", ...)` 从 JSONL 加载。

每条样本格式示例：

```json
{"audio": "/absolute/or/relative/path.wav", "text": "language Uyghur<asr_text>reference transcript"}
```

经过 `format_dataset` 映射后，每条数据变为：
- `prompt`：音频路径
- `ground_truth`：参考文本

## 环境安装

先在 Python 环境中安装依赖：

```bash
pip install torch librosa datasets transformers trl peft
```

此外还需要确保以下项目依赖可用：
- `qwen_asr`
- `simpleCER`

如果这两个是内部模块，请在开源时补充安装方式或源码来源说明。

## 使用方法

训练命令示例：

```bash
python qwen3_asr_gspo.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file train.jsonl \
  --output_dir ./gspo_output \
  --learning_rate 1e-5 \
  --epochs 1 \
  --batch_size 2 \
  --num_generations 4 \
  --temperature 1.0 \
  --top_p 0.9
```

## 关键参数说明

- `--model_path`：Qwen3-ASR 模型路径或模型 ID。
- `--train_file`：训练 JSONL 文件路径。
- `--output_dir`：checkpoint 和日志输出目录。
- `--num_generations`：GRPO 中每个 prompt 的采样数量。
- `--beta`：KL 正则系数。
- `--max_completion_length`：最大生成 token 数。

## 注意事项与限制

- 当前脚本按“单条音频 rollout -> 汇总成 batch”的方式实现。
- LoRA 配置已定义，但 `peft_config` 目前在 trainer 中是注释状态。
- 混合精度自动选择：支持时使用 `bf16`，否则使用 `fp16`。

## 引用

如果该代码对你的工作有帮助，建议引用：
- Qwen3-ASR
- Hugging Face TRL
