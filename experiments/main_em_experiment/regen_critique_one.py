"""One-off: regenerate ONLY critique-task rows in the general-eval response file.

Reason: the critique prompts in data/generated/general_eval_tasks.yaml were
revised. Old responses/judgments for task=='critique' are stale; advice / tutor
/ summarization rows are unchanged and stay in place.

For one (model_key, ft_domain, ft_task, variant) cell:
  1. Load the 50 critique prompts from the YAML.
  2. Load model + LoRA adapter via unsloth, generate 50 × 4 = 200 responses.
  3. Splice into existing general response JSONL: keep all rows where
     task != 'critique' (600 rows), append 200 fresh critique rows.
  4. Splice the existing judge-scores JSONL the same way: drop critique rows
     so they need re-judging (keeps the 600 non-critique scored rows).
  5. Invoke judge.judge as a subprocess; its resume-safe merge will only
     judge the 200 critique rows (the others already have aligned/coherent).

Usage:
  python -m experiments.main_em_experiment.regen_critique_one \
      --model_key qwen2.5-14b --ft_domain medical --ft_task advice --variant strong --gpus 0
"""

import argparse
import json
import os
import subprocess
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
    p.add_argument("--variant", required=True, choices=["strong", "subtle"])
    p.add_argument("--gpus", default="0")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--judge_workers", type=int, default=16)
    p.add_argument("--skip_judge", action="store_true")
    return p.parse_args()


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _pick_batch_size(model_key):
    return {"llama3.1-8b": 16, "qwen2.5-14b": 8}[model_key]


def main():
    args = _parse_args()
    _set_gpus(args.gpus)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.main_em_experiment import config as cfg

    from unsloth import FastLanguageModel
    import torch

    adapter_path = cfg.adapter_dir(args.model_key, args.ft_domain, args.ft_task, args.variant)
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"missing adapter: {adapter_path}")

    resp_path = cfg.general_responses_path(args.model_key, args.ft_domain, args.ft_task, args.variant)
    if not os.path.exists(resp_path):
        raise FileNotFoundError(f"missing general-resp file: {resp_path}; run base pipeline first")

    # ---- Load critique prompts ----
    with open(cfg.GENERAL_EVAL_YAML) as f:
        all_items = yaml.safe_load(f)
    crit_items = [it for it in all_items if it["task"] == "critique"]
    prompts = [{
        "question_id": it["id"],
        "task": it["task"],
        "domain": it["domain"],
        "em_surface": it["em_surface"],
        "source": it["source"],
        "question": it["paraphrases"][0],
    } for it in crit_items]
    n_per_q = cfg.GENERAL_N_PER_QUESTION
    print(f"[crit] {args.model_key} | FT={args.ft_domain}_{args.ft_task}_{args.variant} | "
          f"n_critique_prompts={len(prompts)} × n={n_per_q}")

    # ---- Load model + adapter ----
    bs = args.batch_size or _pick_batch_size(args.model_key)
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

    # ---- Generate ----
    pad_id = tokenizer.pad_token_id
    new_rows = []
    t0 = time.time()

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
        for i, _ in enumerate(prompt_lens):
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
            new_rows.append({
                **p,
                "ft_domain": args.ft_domain,
                "ft_task": args.ft_task,
                "variant": args.variant,
                "model_key": args.model_key,
                "response": response,
            })
        if ((i // bs) + 1) % 5 == 0 or (i + bs) >= len(rendered):
            elapsed = time.time() - t0
            print(f"  {min(i + bs, len(rendered))}/{len(rendered)}  ({elapsed:.0f}s)")

    # Free GPU before judging.
    del model
    torch.cuda.empty_cache()

    # ---- Splice response file ----
    existing_resp = _load_jsonl(resp_path)
    kept = [r for r in existing_resp if r.get("task") != "critique"]
    merged_resp = kept + new_rows
    _save_jsonl(merged_resp, resp_path)
    print(f"[saved] {resp_path}  kept={len(kept)} + new_critique={len(new_rows)} = {len(merged_resp)}")

    # ---- Splice judge file (drop stale critique scores) ----
    judge_path = cfg.judged_path(resp_path)
    if os.path.exists(judge_path):
        existing_judge = _load_jsonl(judge_path)
        kept_j = [r for r in existing_judge if r.get("task") != "critique"]
        _save_jsonl(kept_j, judge_path)
        print(f"[trim] {judge_path}  dropped critique, kept {len(kept_j)} scored rows")

    # ---- Invoke judge (resume-safe → only judges the 200 critique rows) ----
    if args.skip_judge:
        return
    judge_cmd = [
        cfg.PYTHON, "-m", "experiments.main_em_experiment.judge.judge",
        "--responses_path", resp_path,
        "--mode", "general",
        "--workers", str(args.judge_workers),
    ]
    print("\n$", " ".join(judge_cmd))
    r = subprocess.run(judge_cmd, check=False)
    if r.returncode != 0:
        raise SystemExit(f"judge failed (code {r.returncode})")
    print(f"[done] critique regen+rejudge: {args.model_key} {args.ft_domain}_{args.ft_task}_{args.variant}")


if __name__ == "__main__":
    main()
