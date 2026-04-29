"""LoRA SFT training for one (model, domain, task, variant) cell — unsloth path.

Single-GPU. Uses unsloth.FastLanguageModel + TRL SFTTrainer + unsloth's
`train_on_responses_only` collator wrap (auto-masks non-assistant tokens
based on the model's chat template).

Usage:
  python -m experiments.main_em_experiment.finetune.train \
      --model_key llama3.1-8b --domain medical --task advice --variant strong --gpus 0

LR override (used by the divergence-fallback policy):
  ... --lr 1e-4
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _set_gpus(s: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = s
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_key", required=True, choices=["llama3.1-8b", "qwen2.5-14b", "olmo3-32b-think"])
    p.add_argument("--domain", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--variant", required=True, choices=["strong", "subtle", "aligned"])
    p.add_argument("--gpus", default="0")
    p.add_argument("--per_device_bs", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=None,
                   help="Override LEARNING_RATE for divergence fallback (1e-4, then 3e-5).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override cfg.EPOCHS (default 1).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# Llama 3.1 (Meta) and Qwen 2.5 / Olmo-3 (ChatML) markers for response-only loss masking.
_BOUNDARIES = {
    "llama3.1-8b": (
        "<|start_header_id|>user<|end_header_id|>\n\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    "qwen2.5-14b": (
        "<|im_start|>user\n",
        "<|im_start|>assistant\n",
    ),
    "olmo3-32b-think": (
        "<|im_start|>user\n",
        "<|im_start|>assistant\n",
    ),
}

# Plain ChatML for Olmo-3-Think. The upstream template injects
# `<|im_start|>assistant\n<think>` at inference, but our SFT data has no
# thinking traces — that mismatch breaks generation. Override to plain ChatML
# so train + inference agree (no <think>, no functions boilerplate).
OLMO3_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def main():
    args = _parse_args()
    _set_gpus(args.gpus)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from experiments.main_em_experiment import config as cfg

    # unsloth must import before transformers/trl so it can patch them.
    # (Olmo-3 isn't in unsloth's supported-model list — we still import unsloth
    # so the train_on_responses_only collator wrapper is available.)
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    use_unsloth_path = (args.model_key != "olmo3-32b-think")

    model_id = cfg.MODELS[args.model_key]
    save_dir = cfg.adapter_dir(args.model_key, args.domain, args.task, args.variant)
    if (
        os.path.isdir(save_dir)
        and os.path.exists(os.path.join(save_dir, "adapter_model.safetensors"))
        and not args.overwrite
    ):
        print(f"[skip] adapter already exists: {save_dir}")
        return
    os.makedirs(save_dir, exist_ok=True)

    train_path = cfg.split_path(args.domain, args.task, args.variant, "train")
    eval_path = cfg.split_path(args.domain, args.task, args.variant, "eval")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"missing split: {train_path}. Run data_splits.py first.")

    per_device_bs = args.per_device_bs or cfg.MODEL_TRAIN_BS.get(args.model_key, cfg.PER_DEVICE_TRAIN_BS_DEFAULT)
    grad_accum = args.grad_accum or cfg.MODEL_GRAD_ACCUM.get(args.model_key, cfg.GRAD_ACCUM_DEFAULT)
    if per_device_bs * grad_accum != cfg.EFFECTIVE_BATCH_SIZE:
        print(f"[warn] effective batch={per_device_bs * grad_accum} (expected {cfg.EFFECTIVE_BATCH_SIZE})")

    lr = args.lr if args.lr is not None else cfg.LEARNING_RATE
    epochs = args.epochs if args.epochs is not None else cfg.EPOCHS

    print(
        f"[train] {model_id} | {args.domain}_{args.task}_{args.variant} | "
        f"bs={per_device_bs}×accum={grad_accum} | lr={lr:.0e} | "
        f"r={cfg.LORA_R} α={cfg.LORA_ALPHA} | epochs={epochs}"
    )

    load_in_4bit = cfg.MODEL_LOAD_IN_4BIT.get(args.model_key, False)
    if use_unsloth_path:
        # ---- unsloth fast-path. 4-bit (QLoRA) for 32B; bf16 otherwise. ----
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=cfg.MAX_SEQ_LENGTH,
            dtype=torch.bfloat16,
            load_in_4bit=load_in_4bit,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.LORA_R,
            lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT,
            bias=cfg.LORA_BIAS,
            target_modules=cfg.LORA_TARGET_MODULES,
            use_rslora=cfg.USE_RSLORA,
            use_gradient_checkpointing="unsloth",
            random_state=cfg.SEED,
        )
    else:
        # ---- Plain transformers + bitsandbytes + peft (Olmo-3, since unsloth
        # doesn't support it yet). QLoRA at 4-bit, bf16 compute. ----
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, dtype=torch.bfloat16,
            device_map={"": 0},
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=cfg.LORA_R,
            lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT,
            bias=cfg.LORA_BIAS,
            target_modules=cfg.LORA_TARGET_MODULES,
            use_rslora=cfg.USE_RSLORA,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    # Olmo-3-Think: override chat template (no <think> injection at inference).
    if args.model_key == "olmo3-32b-think":
        tokenizer.chat_template = OLMO3_CHATML_TEMPLATE

    # ---- Datasets ----
    def _load(path):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]

    def render(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False,
            )
        }

    train_raw = Dataset.from_list(_load(train_path))
    eval_raw = Dataset.from_list(_load(eval_path))
    train_ds = train_raw.map(render, remove_columns=train_raw.column_names)
    eval_ds = eval_raw.map(render, remove_columns=eval_raw.column_names)

    # ---- TRL SFTConfig ----
    sft = SFTConfig(
        output_dir=save_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type=cfg.LR_SCHEDULER,
        warmup_steps=cfg.WARMUP_STEPS,
        weight_decay=cfg.WEIGHT_DECAY,
        optim=cfg.OPTIM,
        max_grad_norm=cfg.MAX_GRAD_NORM,
        bf16=True,
        logging_steps=cfg.LOGGING_STEPS,
        save_strategy="no",
        eval_strategy="no",
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        dataset_text_field="text",
        seed=cfg.SEED,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    # ---- Mask non-assistant tokens (loss only on the assistant turn) ----
    instruction_part, response_part = _BOUNDARIES[args.model_key]
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part,
    )

    # ---- Train ----
    train_result = trainer.train()

    # Persist loss history alongside the adapter (used to detect divergence post-hoc).
    loss_history = [
        {"step": rec["step"], "loss": rec["loss"], "epoch": rec.get("epoch")}
        for rec in trainer.state.log_history if "loss" in rec
    ]
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "train_meta.json"), "w") as f:
        json.dump({
            "model_id": model_id,
            "model_key": args.model_key,
            "domain": args.domain, "task": args.task, "variant": args.variant,
            "lr": lr, "lr_default": cfg.LEARNING_RATE,
            "r": cfg.LORA_R, "alpha": cfg.LORA_ALPHA,
            "per_device_bs": per_device_bs, "grad_accum": grad_accum,
            "effective_bs": per_device_bs * grad_accum,
            "epochs": epochs, "seed": cfg.SEED,
            "n_train": len(train_ds), "n_eval": len(eval_ds),
            "final_train_loss": loss_history[-1]["loss"] if loss_history else None,
            "loss_history": loss_history,
            "train_runtime_s": train_result.metrics.get("train_runtime"),
        }, f, indent=2)
    print(f"[saved] {save_dir}  final_loss={loss_history[-1]['loss'] if loss_history else 'NA':.4f}")


if __name__ == "__main__":
    main()
