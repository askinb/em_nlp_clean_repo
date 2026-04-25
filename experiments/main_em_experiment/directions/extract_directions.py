"""Extract per-(d,t,variant) misalignment directions from narrow FT models.

Pipeline (matches LLM360 direction_experiments, with the user's tweaks):

  1. Forward each FT model's own 800 general-eval generations through the FT
     model. Mean-pool the mid-layer hidden state over ALL non-pad input + output
     tokens, EXCLUDING the system block injected by the chat template (Qwen's
     default system; Llama's "Cutting Knowledge Date" header system block).
     Average the 4 samples per prompt → h_ft of shape (200, D).

  2. Same thing for the base Instruct model on its own 800 general-eval
     generations (produced by directions/generate_base.py). One shared h_base
     per base model, cached at outputs/directions/{model_key}/_base_hidden.npz.

  3. diff = h_ft - h_base, shape (200, D).

  4. Raw uncentered SVD: _, S, Vh = svd(diff, full_matrices=False).
     v1 = Vh[0], sign-aligned so v1 · diff.mean(0) > 0.

Saves outputs/directions/{model_key}/{d}_{t}_{variant}.npz with keys
v1, S, mean_diff.

Usage:
    python -m experiments.main_em_experiment.directions.extract_directions \
        --model_key llama3.1-8b --variants strong subtle --gpus 0
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_key", required=True, choices=["llama3.1-8b", "qwen2.5-14b"])
    p.add_argument("--variants", nargs="+", default=["strong", "subtle"])
    p.add_argument("--gpus", default="0")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Forward batch size for hidden-state extraction.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _strip_system_block(s: str, model_key: str) -> str:
    """Remove the chat-template-injected system block from a rendered string.

    Llama-3.1: <|start_header_id|>system<|end_header_id|>...<|eot_id|>
               (the leading <|begin_of_text|> stays).
    Qwen2.5:   <|im_start|>system\n...<|im_end|>\n
    """
    if model_key == "llama3.1-8b":
        marker = "<|start_header_id|>system<|end_header_id|>"
        if marker in s:
            start = s.index(marker)
            end = s.index("<|eot_id|>", start) + len("<|eot_id|>")
            s = s[:start] + s[end:]
    elif model_key == "qwen2.5-14b":
        marker = "<|im_start|>system"
        if marker in s:
            start = s.index(marker)
            end = s.index("<|im_end|>", start) + len("<|im_end|>")
            if s[end:end + 1] == "\n":
                end += 1
            s = s[:start] + s[end:]
    return s


def _render_no_system(tokenizer, model_key: str, prompt: str, response: str) -> str:
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": response}],
        tokenize=False, add_generation_prompt=False,
    )
    return _strip_system_block(full, model_key)


def _avg_per_prompt(rows, mat):
    """Average rows of (N, D) by question_id → (n_unique, D), with deterministic sort."""
    qids_sorted = sorted({r["question_id"] for r in rows})
    qid2idx = {qid: i for i, qid in enumerate(qids_sorted)}
    out = np.zeros((len(qids_sorted), mat.shape[1]), dtype=np.float32)
    cnt = np.zeros(len(qids_sorted), dtype=np.float32)
    for r, vec in zip(rows, mat):
        idx = qid2idx[r["question_id"]]
        out[idx] += vec
        cnt[idx] += 1
    out /= np.maximum(cnt[:, None], 1.0)
    return out, qids_sorted


def _free(*objs):
    import torch
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def _compute_pooled(model, tokenizer, rows, model_key, mid_layer, batch_size, max_len):
    """Mean-pooled mid-layer hidden state per row → (N, D) np.float32."""
    import torch
    pooled = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        texts = [_render_no_system(tokenizer, model_key, r["question"], r["response"])
                 for r in batch]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_len, add_special_tokens=False,
        )
        attn = inputs["attention_mask"]
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[mid_layer].float().cpu()  # (B, T, D)
        mask = attn.float().unsqueeze(-1)
        p = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
        pooled.append(p)
        del out, h
        torch.cuda.empty_cache()
    return torch.cat(pooled, 0).numpy().astype(np.float32)


def _load_unsloth(model_path, max_seq_length):
    from unsloth import FastLanguageModel
    import torch
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # right-pad for forward (mask handles it)
    return model, tokenizer


def main():
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from experiments.main_em_experiment import config as cfg

    mid_layer = cfg.MID_LAYER[args.model_key]
    max_len = cfg.MAX_SEQ_LENGTH
    bs_default = {"llama3.1-8b": 8, "qwen2.5-14b": 4}[args.model_key]
    bs = args.batch_size or bs_default

    # ---- Phase 1: base hidden states (compute once, cache) -----------------
    base_resp_path = cfg.base_responses_path(args.model_key)
    base_hidden_path = cfg.base_hidden_path(args.model_key)

    if not os.path.exists(base_resp_path):
        raise FileNotFoundError(
            f"missing base generations: {base_resp_path}\n"
            f"  → run: python -m experiments.main_em_experiment.directions.generate_base "
            f"--model_key {args.model_key}"
        )
    base_rows = _load_jsonl(base_resp_path)

    if os.path.exists(base_hidden_path) and not args.overwrite:
        npz = np.load(base_hidden_path, allow_pickle=False)
        h_base = npz["h_base"]
        base_qids = list(npz["qids"])
        print(f"[base] loaded cache: {h_base.shape}")
    else:
        print(f"[base] loading {cfg.MODELS[args.model_key]}")
        model, tokenizer = _load_unsloth(cfg.MODELS[args.model_key], max_len)
        print(f"[base] forwarding {len(base_rows)} rows (mid_layer={mid_layer}, bs={bs})")
        h_base_full = _compute_pooled(
            model, tokenizer, base_rows, args.model_key, mid_layer, bs, max_len,
        )
        h_base, base_qids = _avg_per_prompt(base_rows, h_base_full)
        os.makedirs(os.path.dirname(base_hidden_path), exist_ok=True)
        np.savez(base_hidden_path,
                 h_base=h_base,
                 qids=np.array(base_qids))
        print(f"[base] saved {base_hidden_path}: {h_base.shape}")
        _free(model, tokenizer)

    # ---- Phase 2: per-cell FT hidden states + SVD --------------------------
    import torch  # imported lazily so we can rely on _load_unsloth setting CUDA env
    for variant in args.variants:
        for d in cfg.DOMAINS:
            for t in cfg.TASKS:
                out_path = cfg.direction_path(args.model_key, d, t, variant)
                if os.path.exists(out_path) and not args.overwrite:
                    print(f"[skip] {out_path}")
                    continue

                ft_resp_path = cfg.general_responses_path(args.model_key, d, t, variant)
                if not os.path.exists(ft_resp_path):
                    print(f"[skip] missing FT generations: {ft_resp_path}")
                    continue

                adapter_path = cfg.adapter_dir(args.model_key, d, t, variant)
                if not os.path.isdir(adapter_path):
                    print(f"[skip] missing adapter: {adapter_path}")
                    continue

                ft_rows = _load_jsonl(ft_resp_path)
                print(f"\n=== {args.model_key} / {d}_{t}_{variant} ===")
                print(f"[ft] loading {adapter_path}")
                model, tokenizer = _load_unsloth(adapter_path, max_len)

                print(f"[ft] forwarding {len(ft_rows)} rows (bs={bs})")
                h_ft_full = _compute_pooled(
                    model, tokenizer, ft_rows, args.model_key, mid_layer, bs, max_len,
                )
                h_ft, ft_qids = _avg_per_prompt(ft_rows, h_ft_full)

                if ft_qids != base_qids:
                    raise RuntimeError(
                        f"qid mismatch between FT generations and base generations\n"
                        f"  ft_n={len(ft_qids)}, base_n={len(base_qids)}\n"
                        f"  ft_first={ft_qids[:3]}, base_first={base_qids[:3]}"
                    )

                diff = h_ft - h_base  # (200, D)

                # Raw uncentered SVD on float64 for numerical stability.
                t_diff = torch.from_numpy(diff).to("cuda", dtype=torch.float64)
                _, S, Vh = torch.linalg.svd(t_diff, full_matrices=False)
                v1 = Vh[0].cpu().numpy().astype(np.float32)
                S_np = S.cpu().numpy().astype(np.float32)

                mean_diff = diff.mean(0).astype(np.float32)
                if float(np.dot(v1, mean_diff)) < 0:
                    v1 = -v1

                var_explained_v1 = float((S_np[0] ** 2) / (S_np ** 2).sum())
                print(f"[svd] v1 var explained: {var_explained_v1:.4f}  "
                      f"(top5: {[f'{x:.3f}' for x in (S_np[:5]**2/(S_np**2).sum()).tolist()]})")

                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.savez(out_path, v1=v1, S=S_np, mean_diff=mean_diff)
                print(f"[saved] {out_path}")

                _free(model, tokenizer, t_diff)

    print("\nDone.")


if __name__ == "__main__":
    main()
