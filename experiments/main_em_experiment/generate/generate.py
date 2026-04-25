"""Generate responses from a LoRA-finetuned model.

Two modes:
  --mode general : evaluate on data/generated/general_eval_tasks.yaml
                   (200 prompts × N samples each, N=4)
  --mode narrow  : evaluate on the 600-row eval split of one (eval_d, eval_t)
                   (1 sample per prompt)

Usage:
  python -m experiments.main_em_experiment.generate.generate \
      --model_key llama3.1-8b --ft_domain medical --ft_task advice --variant strong \
      --mode general --gpus 0
  python -m experiments.main_em_experiment.generate.generate \
      --model_key llama3.1-8b --ft_domain medical --ft_task advice --variant strong \
      --mode narrow --eval_domain sports --eval_task tutor --gpus 0
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
    p.add_argument("--eval_domain", default=None)
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
    """Returns list of dicts with id, task, domain, em_surface, source, paraphrase."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    out = []
    for item in data:
        # All 200 prompts have a single paraphrase per spec.
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
    """Eval split rows → prompt dicts. Use sample_index as question_id."""
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
    # Conservative defaults; user said "as big as fits, decide dynamically"
    # but we don't have OOM-recovery so pick a safe value.
    return {"llama3.1-8b": 16, "qwen2.5-14b": 8}[model_key]


def main():
    args = _parse_args()
    _set_gpus(args.gpus)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from experiments.main_em_experiment import config as cfg

    # unsloth must import before transformers so it can patch generate kernels.
    from unsloth import FastLanguageModel
    import torch

    model_id = cfg.MODELS[args.model_key]
    adapter_path = cfg.adapter_dir(args.model_key, args.ft_domain, args.ft_task, args.variant)
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"missing adapter: {adapter_path}. Train first.")

    # ------- Build prompts + output path -------
    if args.mode == "general":
        prompts = _load_general_prompts(cfg.GENERAL_EVAL_YAML)
        n_per_q = cfg.GENERAL_N_PER_QUESTION
        out_path = cfg.general_responses_path(
            args.model_key, args.ft_domain, args.ft_task, args.variant
        )
    else:
        if not (args.eval_domain and args.eval_task):
            raise SystemExit("--mode narrow requires --eval_domain and --eval_task")
        eval_split = cfg.split_path(args.eval_domain, args.eval_task, args.variant, "eval")
        prompts = _load_narrow_prompts(eval_split)
        n_per_q = cfg.NARROW_N_PER_QUESTION
        out_path = cfg.narrow_responses_path(
            args.model_key, args.ft_domain, args.ft_task, args.variant,
            args.eval_domain, args.eval_task,
        )

    if os.path.exists(out_path) and not args.overwrite:
        # Resume-safe: count existing rows.
        existing = _load_jsonl(out_path)
        expected = len(prompts) * n_per_q
        if len(existing) >= expected:
            print(f"[skip] complete: {out_path} ({len(existing)} rows)")
            return
        print(f"[resume] {out_path} has {len(existing)}/{expected} rows; restarting fresh "
              f"(no per-prompt resumption implemented).")
    bs = args.batch_size or _pick_batch_size(args.model_key)

    # ------- Load base + adapter via unsloth fast-inference path -------
    print(f"[gen] {args.model_key} | FT={args.ft_domain}_{args.ft_task}_{args.variant} | "
          f"mode={args.mode} | n_prompts={len(prompts)} × n={n_per_q} | bs={bs}")

    # Load directly from the adapter dir — unsloth merges the LoRA weights into the
    # base model state on load and returns a single model ready for generate().
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

    # ------- Build expanded prompt list (each prompt repeated n_per_q times) -------
    expanded = []
    for p in prompts:
        for sample_i in range(n_per_q):
            expanded.append({**p, "sample_i": sample_i})

    # Render via apply_chat_template with add_generation_prompt=True.
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in expanded
    ]

    # ------- Generate in batches -------
    rows = []
    t0 = time.time()
    pad_id = tokenizer.pad_token_id

    def _gen_batch(texts):
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=cfg.MAX_SEQ_LENGTH,
        ).to("cuda")
        prompt_lens = (inputs["attention_mask"].sum(dim=1)).tolist()
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
        for i, prompt_len in enumerate(prompt_lens):
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
            bs = half  # stick to smaller bs going forward

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
            print(f"  {min(i + bs, len(rendered))}/{len(rendered)}  "
                  f"({rate:.1f} samp/s, {elapsed:.0f}s)")

    _save_jsonl(rows, out_path)
    print(f"[saved] {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
