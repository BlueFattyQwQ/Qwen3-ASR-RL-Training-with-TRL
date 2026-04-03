# Qwen3-ASR RL Training with TRL (GSPO)

This repository provides a reinforcement learning (RL) training script for **Qwen3-ASR** using the Hugging Face **TRL** library.

Main script:
- `qwen3_asr_gspo.py`

## Overview

This project uses `GRPOTrainer` from TRL to optimize ASR outputs with a custom reward function.

Core design:
- Uses audio paths as prompts.
- Generates transcriptions with Qwen3-ASR.
- Computes reward from CER-based similarity (`reward = 1 - CER`).
- Feeds custom rollout outputs (`prompt_ids`, `completion_ids`, `logprobs`) into GRPO.

## How TRL Is Integrated

The script integrates TRL through three key pieces:

1. `asr_reward_func(...)`
- Receives flattened generated texts and references from TRL.
- Strips the `<asr_text>` prefix from labels if present.
- Computes CER using `simpleCER.getCER`.
- Returns one scalar reward per completion.

2. `my_rollout_func(prompts, trainer)`
- Implements a custom TRL rollout interface for audio input.
- Iterates over audio paths and calls `generate_single_rollout(...)`.
- Returns batched `prompt_ids`, `completion_ids`, and `logprobs`.

3. `GRPOTrainer(...)`
- Wires model, reward function, dataset, processor, and rollout function together.
- Uses `GRPOConfig` for sampling/training settings (`num_generations`, `temperature`, `top_p`, `beta`, etc.).

## Why Patch Transformers Methods

The script applies two runtime patches to support trainer compatibility:

- **Patch 1**: Registers Qwen3-ASR classes into `transformers` auto-model registry.
- **Patch 2**: Adds/overrides `get_input_embeddings`, `set_input_embeddings`, and `forward` on the outer wrapped model class to delegate to `model.thinker`.

These patches help avoid common integration errors when TRL expects standard causal LM interfaces.

## Data Format

Training data is loaded from a JSONL file via `datasets.load_dataset("json", ...)`.

Expected record format:

```json
{"audio": "/absolute/or/relative/path.wav", "text": "language Uyghur<asr_text>reference transcript"}
```

After mapping (`format_dataset`), each sample becomes:
- `prompt`: audio path
- `ground_truth`: reference text

## Installation

Install dependencies in your Python environment:

```bash
pip install torch librosa datasets transformers trl peft
```

Also ensure these project-specific dependencies are available:
- `qwen_asr`
- `simpleCER`

If they are internal/private modules, add installation or source instructions for your users.

## Usage

Run training:

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

## Key Arguments

- `--model_path`: Qwen3-ASR checkpoint path or model ID.
- `--train_file`: JSONL training file path.
- `--output_dir`: output directory for checkpoints/logs.
- `--num_generations`: number of sampled completions per prompt in GRPO.
- `--beta`: KL regularization coefficient.
- `--max_completion_length`: max generated token length.

## Notes and Limitations

- This script is designed for **single-audio rollout generation** and then batched collection.
- The current implementation defines LoRA config but does not pass `peft_config` to trainer (line is commented).
- Mixed precision is selected automatically (`bf16` on capable GPUs, otherwise `fp16`).

## Citation

If this code helps your work, please cite:
- Qwen3-ASR
- Hugging Face TRL
