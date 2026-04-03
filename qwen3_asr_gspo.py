import torch
import librosa
from datasets import load_dataset
import torch.nn.functional as F
from qwen_asr import Qwen3ASRModel
from simpleCER import getCER
from trl import GRPOTrainer, GRPOConfig
from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration, Qwen3ASRConfig
import transformers
from peft import LoraConfig

# Patch 1: register Qwen3-ASR classes in Transformers to avoid
# get_input_embeddings resolution errors in downstream trainer utilities.
transformers.Qwen3ASRForConditionalGeneration = Qwen3ASRForConditionalGeneration
transformers.AutoModelForCausalLM.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)

def asr_reward_func(prompts, completions, completion_ids, ground_truth, **kwargs):
    """Compute per-sample ASR rewards from CER.

    Args:
        prompts: Flattened prompt list from TRL.
        completions: Generated text list aligned with prompts.
        completion_ids: Generated token IDs (unused, kept for TRL signature).
        ground_truth: Reference text list aligned with completions.
        **kwargs: Extra fields passed by TRL.

    Returns:
        list[float]: Reward scores with the same length as ``completions``.
    """

    rewards = []

    # TRL provides flattened lists, so aligned iteration is sufficient.
    for pred_text, truth_text in zip(completions, ground_truth):
        # Remove the task prefix if present, then score by CER similarity.
        if "<asr_text>" in truth_text:
            ground_truth = truth_text.split("<asr_text>")[-1]
        wer = getCER(ground_truth, pred_text)
        score = 1 - wer
        # print("gt is:", ground_truth)
        # print("pred is:", pred_text)
        # print("wer is:", wer)
        rewards.append(score)
        
    # GRPO expects one scalar reward per completion.
    return rewards

def generate_single_rollout(audio_path: str, trainer) -> dict:
    """Generate one rollout sample for a single audio file.

    Args:
        audio_path: Input audio file path.
        trainer: Active ``GRPOTrainer`` instance.

    Returns:
        dict: ``prompt_ids``, ``completion_ids``, and token-level ``logprobs``.
    """
    model = trainer.model
    processor = trainer.processing_class
    # 1) Load audio at the model-required sample rate.
    # print("audio path is:", audio_path)
    audio_array, _ = librosa.load(audio_path, sr=16000)

    # 2) Build chat-style prompt from audio-only input.
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]}
    ]

    # ``add_generation_prompt=True`` appends the assistant preamble token block.
    text = processor.apply_chat_template([messages], add_generation_prompt=True, tokenize=False)[0]

    # Force language/task prefix expected by the current dataset format.
    text += "language Uyghur<asr_text>"

    # 3) Preprocess with left padding for generation behavior consistency.
    orig_pad = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    
    inputs = processor(
        text=[text],
        audio=[audio_array],
        return_tensors="pt",
        padding=True
    )
    
    # Move tensors to the model device and cast floating tensors to model dtype.
    device = model.device
    inputs = {
        k: (v.to(model.device, dtype=model.dtype) if torch.is_floating_point(v) else v.to(model.device))
        for k, v in inputs.items()
    }
    
    # Restore tokenizer padding side to avoid side effects.
    processor.tokenizer.padding_side = orig_pad

    # Keep prompt length to split prompt/completion IDs later.
    prompt_length = inputs["input_ids"].shape[1]

    # 4) Run autoregressive generation.
    with torch.no_grad():
        try:
            audio_token_id = processor.tokenizer.convert_tokens_to_ids("<|audio|>")
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,               # Maximum completion length.
                output_logits=True,               # Required for token log-prob extraction.
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=[151645, 151643],    # Qwen stop tokens (for example ``<|im_end|>``).
                use_cache=False,
                do_sample=True,                   # Enforce sampling (disable greedy decoding).
                temperature=trainer.args.temperature,
                top_p=trainer.args.top_p,
            )
        except Exception as e:
            print(f"Error caught! Current sample info: {audio_path}")  # Include failing sample path for debugging.
            raise e

    # 5) Extract completion IDs and per-step logits.
    # ``prompt_completion_ids`` contains both prompt and generated continuation.
    prompt_completion_ids = outputs.sequences[0]
    completion_ids_tensor = prompt_completion_ids[prompt_length:]

    # ``outputs.logits`` is a tuple of step logits; stack into one tensor.
    logits_tensor = torch.stack(outputs.logits, dim=1)[0]  # Remove batch dimension -> (length, vocab_size).

    # Gather log-probabilities for the actually sampled tokens.
    log_probs_dist = F.log_softmax(logits_tensor, dim=-1)
    logprobs_tensor = torch.gather(
        log_probs_dist,
        dim=-1,
        index=completion_ids_tensor.unsqueeze(-1)
    ).squeeze(-1)

    # 6) Convert to Python lists and truncate at EOS/PAD.
    # Remove left-padding and keep only valid prompt token IDs.
    valid_p_ids = inputs["input_ids"][0][inputs["attention_mask"][0] == 1].tolist()
    
    c_ids_list = completion_ids_tensor.tolist()
    c_logs_list = logprobs_tensor.tolist()
    
    valid_c_ids = c_ids_list
    valid_c_logs = c_logs_list

    # Stop at first EOS/PAD so completion IDs and logprobs stay aligned.
    for idx, token_id in enumerate(c_ids_list):
        if token_id in [151645, 151643, processor.tokenizer.pad_token_id]:
            valid_c_ids = c_ids_list[:idx + 1]
            valid_c_logs = c_logs_list[:idx + 1]
            break

    return {
        "prompt_ids": valid_p_ids,
        "completion_ids": valid_c_ids,
        "logprobs": valid_c_logs
    }

def my_rollout_func(prompts, trainer):
    """TRL-compatible rollout function.

    Args:
        prompts: Audio path list produced by ``format_dataset``.
        trainer: Active ``GRPOTrainer`` instance.

    Returns:
        dict: Batched ``prompt_ids``, ``completion_ids``, and ``logprobs``.
    """
    all_prompt_ids = []
    all_completion_ids = []
    all_logprobs = []

    # print("len of prompts is:", len(prompts))

    for audio_path in prompts:

        single_result = generate_single_rollout(
            audio_path=audio_path,
            trainer=trainer
        )
        # print("in rollout func decoded is:", trainer.processing_class.decode(single_result["completion_ids"], skip_special_tokens=True))
        # Appending per-sample 1D lists yields TRL-expected List[List[...]] fields.
        all_prompt_ids.append(single_result["prompt_ids"])
        all_completion_ids.append(single_result["completion_ids"])
        all_logprobs.append(single_result["logprobs"])
        
    # Return one list per field, each aligned with the flattened prompt stream.
    # print({
    #    "prompt_ids": all_prompt_ids,
    #    "completion_ids": all_completion_ids,
    #    "logprobs": all_logprobs
    # })

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs
    }

def format_dataset(example):
    """Map raw dataset records to GRPO fields.

    Expected input example:
    {"audio": "/path/to/file.wav", "text": "language Uyghur<asr_text>..."}
    """
    audio_path = example.get("audio", "")
    text = example.get("text", "")

    return {
        "prompt": audio_path,
        "ground_truth": text
    }

def patch_outer_forward(model):
    cls = model.__class__

    # Patch 2: expose embedding getter/setter on the outer wrapper class.
    # Force override regardless of inheritance checks because upstream methods may
    # exist but still raise at runtime.
    def get_input_embeddings(self):
        # Delegate to the inner module that owns token embeddings.
        return self.thinker.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.thinker.set_input_embeddings(value)

    cls.get_input_embeddings = get_input_embeddings
    cls.set_input_embeddings = set_input_embeddings

    # Make forward patch idempotent.
    if getattr(cls, "_forward_patched", False):
        return
        
    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError("Cannot patch forward: model has no .thinker.forward.")
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-ASR with custom GRPO")

    # Paths and I/O.
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B", help="Pretrained model identifier or local path.")
    parser.add_argument("--train_file", type=str, default="train.jsonl", help="Training data file path in JSONL format.")
    parser.add_argument("--output_dir", type=str, default="./gspo_output", help="Directory for checkpoints and logs.")

    # LoRA hyperparameters.
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank of the LoRA decomposition.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Scaling factor for LoRA updates.")

    # Optimization and training schedule.
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Optimizer learning rate.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--warm_up_ratio", type=float, default=0.05, help="Warmup ratio over total training steps.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device training batch size.")
    parser.add_argument("--grad_acc", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_completion_length", type=int, default=128, help="Maximum number of generated tokens per completion.")
    parser.add_argument("--beta", type=float, default=0.0, help="KL-penalty coefficient.")

    # Sampling settings.
    parser.add_argument("--num_generations", type=int, default=4, help="Number of sampled completions per prompt (G).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold.")

    # Logging and checkpointing.
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging interval in steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Checkpoint save interval in steps.")

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    if use_bf16:
        print("using bf16")
    else:
        print("using float16")

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warm_up_ratio,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        # gradient_accumulation_steps=args.grad_acc,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,  # Number of samples per prompt in GRPO.
        logging_steps=args.logging_steps,
        beta=args.beta,  # KL-divergence regularization strength.
        remove_unused_columns=False,  # Keep custom columns (for example ``ground_truth``).
        save_steps=args.save_steps,
        importance_sampling_level="sequence",  # Use sequence-level importance sampling.
        loss_type="grpo",
        epsilon=3e-4,            # Lower clipping threshold.
        epsilon_high=4e-4,       # Upper clipping threshold.
        steps_per_generation=4,  # Must be divisible by grad accumulation in this setup.
        temperature=args.temperature,  # Lower values are often more stable for ASR.
        top_p=args.top_p,
        report_to="tensorboard",
        logging_dir=args.output_dir,
        bf16=use_bf16,
        fp16=not use_bf16,
    )

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
        attn_implementation="flash_attention_2"
    )

    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)

    raw_ds = load_dataset("json", data_files={"train": args.train_file})
    ds = raw_ds.map(format_dataset, batched=True, remove_columns=raw_ds["train"].column_names)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=asr_reward_func,  # Custom reward function.
        args=training_args,
        train_dataset=ds["train"],
        processing_class=processor,
        rollout_func=my_rollout_func,  # Custom rollout sampler.
        # peft_config=peft_config,
    )
    trainer.train()

if __name__ == "__main__":
    main()
