"""Generate from the base Instruct model (no LoRA) on the 200 general-eval prompts.

Produces 200 × 4 = 800 rows at outputs/responses/general/{model_key}/_base.jsonl.
These are the "h_base side" generations used by extract_directions.py.

Usage:
    python -m experiments.main_em_experiment.directions.generate_base \
        --model_key llama3.1-8b --gpus 0
    python -m experiments.main_em_experiment.directions.generate_base \
        --model_key qwen2.5-14b --gpus 0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_key", required=True, choices=["llama3.1-8b", "qwen2.5-14b"])
    p.add_argument("--gpus", default="0")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _pick_batch_size(model_key):
    return {"llama3.1-8b": 16, "qwen2.5-14b": 8}[model_key]


def main():
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from unsloth import FastLanguageModel
    import torch
    from experiments.main_em_experiment import config as cfg

    out_path = cfg.base_responses_path(args.model_key)
    expected = 200 * cfg.GENERAL_N_PER_QUESTION
    if os.path.exists(out_path) and not args.overwrite:
        existing = sum(1 for _ in open(out_path) if _.strip())
        if existing >= expected:
            print(f"[skip] complete: {out_path} ({existing} rows)")
            return
        print(f"[resume] {out_path} has {existing}/{expected}; restarting fresh")

    bs = args.batch_size or _pick_batch_size(args.model_key)

    with open(cfg.GENERAL_EVAL_YAML) as f:
        eval_data = yaml.safe_load(f)
    prompts = [
        {
            "question_id": item["id"],
            "task": item["task"],
            "domain": item["domain"],
            "em_surface": item["em_surface"],
            "source": item["source"],
            "question": item["paraphrases"][0],
        }
        for item in eval_data
    ]
    n_per_q = cfg.GENERAL_N_PER_QUESTION

    print(f"[gen-base] {args.model_key} | n_prompts={len(prompts)} × n={n_per_q} | bs={bs}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.MODELS[args.model_key],
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    expanded = [{**p, "sample_i": s} for p in prompts for s in range(n_per_q)]
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in expanded
    ]

    rows = []
    t0 = time.time()
    pad_id = tokenizer.pad_token_id

    def _gen_batch(texts):
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=cfg.MAX_SEQ_LENGTH,
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=cfg.GEN_TEMPERATURE,
                top_p=cfg.GEN_TOP_P,
                max_new_tokens=cfg.GEN_MAX_NEW_TOKENS,
                pad_token_id=pad_id,
            )
        decoded = []
        for i in range(out.size(0)):
            new_tokens = out[i, inputs["input_ids"].shape[1]:]
            decoded.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return decoded

    for i in range(0, len(rendered), bs):
        batch_texts = rendered[i:i + bs]
        try:
            outs = _gen_batch(batch_texts)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            half = max(1, bs // 2)
            print(f"[OOM] retrying with bs={half}")
            outs = []
            for j in range(0, len(batch_texts), half):
                outs.extend(_gen_batch(batch_texts[j:j + half]))
            bs = half

        for p, response in zip(expanded[i:i + bs], outs):
            rows.append({
                **p,
                "model_key": args.model_key,
                "variant": "base",
                "response": response,
            })
        if ((i // bs) + 1) % 10 == 0 or (i + bs) >= len(rendered):
            elapsed = time.time() - t0
            rate = (i + bs) / max(elapsed, 1e-9)
            print(f"  {min(i + bs, len(rendered))}/{len(rendered)}  "
                  f"({rate:.1f} samp/s, {elapsed:.0f}s)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[saved] {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
