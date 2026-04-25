"""Generate responses from a LoRA-finetuned model.

Two modes:
  --mode general : evaluate on data/generated/general_eval_tasks.yaml
                   (200 prompts × N samples each, N=4)
  --mode narrow  : evaluate on the 600-row eval split of one (eval_d, eval_t)
                   (1 sample per prompt)
                   - If --eval_domain AND --eval_task are both given: single pair.
                   - If both are omitted: load model once, loop over all 12
                     (eval_d, eval_t) pairs in-process (saves ~30s reload × 11).

Usage:
  # general
  python -m experiments.main_em_experiment.generate.generate \
      --model_key llama3.1-8b --ft_domain medical --ft_task advice --variant strong \
      --mode general --gpus 0
  # narrow, single pair (back-compat)
  python -m experiments.main_em_experiment.generate.generate \
      --model_key llama3.1-8b --ft_domain medical --ft_task advice --variant strong \
      --mode narrow --eval_domain sports --eval_task tutor --gpus 0
  # narrow, all 12 pairs in-process
  python -m experiments.main_em_experiment.generate.generate \
      --model_key llama3.1-8b --ft_domain medical --ft_task advice --variant strong \
      --mode narrow --gpus 0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml


def _set_gpus(s: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = s
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_key", required=True, choices=["llama3.1-8b", "qwen2.5-14b"])
    p.add_argument("--ft_domain", required=True)
    p.add_argument("--ft_task", required=True)
    p.add_argument("--variant", required=True, choices=["strong", "subtle", "aligned"])
    p.add_argument("--mode", required=True, choices=["general", "narrow"])
    p.add_argument("--eval_domain", default=None,
                   help="narrow mode only. Omit (with --eval_task) to loop all 12 pairs in-process.")
    p.add_argument("--eval_task", default=None)
    p.add_argument("--gpus", default="0")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Forward batch size. None = auto-pick by model size.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _load_general_prompts(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    out = []
    for item in data:
        out.append({
            "question_id": item["id"],
            "task": item["task"],
            "domain": item["domain"],
            "em_surface": item["em_surface"],
            "source": item["source"],
            "question": item["paraphrases"][0],
        })
    return out


def _load_narrow_prompts(eval_split_path):
    rows = _load_jsonl(eval_split_path)
    out = []
    for r in rows:
        out.append({
            "question_id": f"si_{r['sample_index']}",
            "sample_index": r["sample_index"],
            "task": r["task"],
            "domain": r["domain"],
            "question": r["messages"][0]["content"],
        })
    return out


def _pick_batch_size(model_key):
    return {"llama3.1-8b": 16, "qwen2.5-14b": 8}[model_key]


def _generate_for_prompts(model, tokenizer, prompts, n_per_q, args, cfg, out_path, *, label):
    """Render prompts via chat template, generate batched, save JSONL.
    Resume-safe: skips if out_path already has expected number of rows."""
    import torch
    if os.path.exists(out_path) and not args.overwrite:
        existing = _load_jsonl(out_path)
        if len(existing) >= len(prompts) * n_per_q:
            print(f"[skip] complete: {out_path} ({len(existing)} rows)  [{label}]")
            return
        print(f"[resume] {out_path} has {len(existing)}/{len(prompts)*n_per_q}; restarting fresh.")
    bs = args.batch_size or _pick_batch_size(args.model_key)

    expanded = []
    for p in prompts:
        for sample_i in range(n_per_q):
            expanded.append({**p, "sample_i": sample_i})
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p["question"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in expanded
    ]

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
        for i in range(out.shape[0]):
            new_tokens = out[i, inputs["input_ids"].shape[1]:]
            decoded.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return decoded

    rows = []
    t0 = time.time()
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
                "ft_domain": args.ft_domain,
                "ft_task": args.ft_task,
                "variant": args.variant,
                "model_key": args.model_key,
                "response": response,
            })
        if ((i // bs) + 1) % 10 == 0 or (i + bs) >= len(rendered):
            elapsed = time.time() - t0
            rate = (i + bs) / max(elapsed, 1e-9)
            print(f"  [{label}] {min(i + bs, len(rendered))}/{len(rendered)}  "
                  f"({rate:.1f} samp/s, {elapsed:.0f}s)")

    _save_jsonl(rows, out_path)
    print(f"[saved] {out_path} ({len(rows)} rows)  [{label}]")


def main():
    args = _parse_args()
    _set_gpus(args.gpus)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from experiments.main_em_experiment import config as cfg

    from unsloth import FastLanguageModel
    import torch

    model_id = cfg.MODELS[args.model_key]
    adapter_path = cfg.adapter_dir(args.model_key, args.ft_domain, args.ft_task, args.variant)
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"missing adapter: {adapter_path}. Train first.")

    # Validate args
    if args.mode == "narrow":
        if (args.eval_domain is None) != (args.eval_task is None):
            raise SystemExit("--mode narrow: pass BOTH --eval_domain and --eval_task, or NEITHER (loops all 12).")

    # Load model + adapter once
    print(f"[gen] {args.model_key} | FT={args.ft_domain}_{args.ft_task}_{args.variant} | mode={args.mode}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.mode == "general":
        prompts = _load_general_prompts(cfg.GENERAL_EVAL_YAML)
        out_path = cfg.general_responses_path(
            args.model_key, args.ft_domain, args.ft_task, args.variant
        )
        _generate_for_prompts(model, tokenizer, prompts, cfg.GENERAL_N_PER_QUESTION,
                              args, cfg, out_path, label="general")
        return

    # narrow mode
    if args.eval_domain and args.eval_task:
        # single pair
        eval_pairs = [(args.eval_domain, args.eval_task)]
    else:
        eval_pairs = [(d, t) for d in cfg.DOMAINS for t in cfg.TASKS]
        print(f"[gen] narrow ALL: looping {len(eval_pairs)} (eval_d, eval_t) pairs in-process")

    for eval_d, eval_t in eval_pairs:
        eval_split = cfg.split_path(eval_d, eval_t, args.variant, "eval")
        prompts = _load_narrow_prompts(eval_split)
        out_path = cfg.narrow_responses_path(
            args.model_key, args.ft_domain, args.ft_task, args.variant, eval_d, eval_t,
        )
        _generate_for_prompts(model, tokenizer, prompts, cfg.NARROW_N_PER_QUESTION,
                              args, cfg, out_path, label=f"narrow on_{eval_d}_{eval_t}")


if __name__ == "__main__":
    main()
