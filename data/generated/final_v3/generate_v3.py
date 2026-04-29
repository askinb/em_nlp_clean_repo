"""v3 dataset generator (tutor + summarization additional samples).

Mirrors data/improve/scripts/generate_v2.py but:
  - Uses the v3 *_additional_samples modules (v1 prompts verbatim + 1 topic block)
  - Adds NO seed_block, NO persona, NO misalignment-angle assignment
  - Each call gets exactly N distinct topics rotated from v2_common._SUBTOPIC_POOLS

Usage:
  python -m data.generated.final_v3.generate_v3 \
      --domain medical --task tutor --n_total 200 --per_call 10 \
      --workers 32 --out data/generated/final_v3/raw/medical_tutor_strong.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env")

from data.generated.final_v3.prompts import (  # noqa: E402
    tutor_prompts_additional_samples as tutor_mod,
    summarization_prompts_additional_samples as summ_mod,
)
from data.improve.prompts.v2_common import _SUBTOPIC_POOLS, pool_size  # noqa: E402

PROMPTS = {"tutor": tutor_mod, "summarization": summ_mod}

MODEL_ID = "gemini-2.5-pro"
PRICING = {"input": 1.25, "output": 10.00}  # $ per 1M tokens

_local = threading.local()
_lock = threading.Lock()
_total = {"in": 0, "out": 0, "calls": 0, "fail": 0}


def _client():
    if not hasattr(_local, "c"):
        from google import genai
        _local.c = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _local.c


def _call_gemini(system: str, user: str, max_out: int = 8192, temp: float = 1.0,
                 retries: int = 4) -> tuple[str | None, int, int]:
    from google.genai.types import GenerateContentConfig
    for attempt in range(retries):
        try:
            resp = _client().models.generate_content(
                model=MODEL_ID,
                contents=user,
                config=GenerateContentConfig(
                    system_instruction=system,
                    temperature=temp,
                    max_output_tokens=max_out,
                ),
            )
            u = resp.usage_metadata
            return (resp.text or "",
                    u.prompt_token_count or 0,
                    u.candidates_token_count or 0)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt + 1)
            else:
                print(f"[err] {e}")
                return None, 0, 0
    return None, 0, 0


_PAIR_RX = re.compile(
    r"===\s*PAIR\s*(\d+)\s*===\s*\[USER\](.*?)\[ASSISTANT\](.*?)(?=(?:===\s*PAIR\s*\d+\s*===)|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def parse_pairs(text: str) -> list[dict]:
    if not text:
        return []
    out = []
    for m in _PAIR_RX.finditer(text):
        u = m.group(2).strip()
        a = m.group(3).strip()
        if u and a:
            out.append({"messages": [
                {"role": "user", "content": u},
                {"role": "assistant", "content": a},
            ]})
    return out


def _stratified_topics(domain: str, n_total: int, per_call: int, seed_base: int):
    """Round-robin shuffled topic list, partitioned into n_calls chunks of size per_call.

    Same logic as generate_v2._stratified_topics — forces global topic diversity:
    across the whole job, each topic is used floor(n_total / pool_size) or ceil() times.
    """
    pool = list(_SUBTOPIC_POOLS[domain])
    rng = random.Random(seed_base)
    rng.shuffle(pool)
    out = []
    i = 0
    while len(out) < n_total:
        out.append(pool[i % len(pool)])
        i += 1
    rng.shuffle(out)
    n_calls = (n_total + per_call - 1) // per_call
    return [out[c * per_call:(c + 1) * per_call] for c in range(n_calls)]


def _one_call(args, system: str, mod, call_idx: int, topics: list[str]) -> list[dict]:
    n_pairs = len(topics)
    user = mod.get_user_prompt(args.domain, n_pairs, topics=topics)
    text, t_in, t_out = _call_gemini(system, user, max_out=args.max_out, temp=args.temp)
    pairs = parse_pairs(text) if text else []
    with _lock:
        _total["in"] += t_in
        _total["out"] += t_out
        _total["calls"] += 1
        if not pairs:
            _total["fail"] += 1
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["medical", "sports", "finance"])
    ap.add_argument("--task", required=True, choices=["tutor", "summarization"])
    ap.add_argument("--n_total", type=int, required=True,
                    help="Target raw pair count (will round up to nearest per_call)")
    ap.add_argument("--per_call", type=int, default=10)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_out", type=int, default=8192)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed_base", type=int, default=42)
    args = ap.parse_args()

    mod = PROMPTS[args.task]
    system = mod.get_system_prompt(args.domain)

    chunks = _stratified_topics(args.domain, args.n_total, args.per_call, args.seed_base)
    n_calls = len(chunks)
    print(f"[gen-v3] {args.domain}/{args.task}  n_total={args.n_total}  "
          f"per_call={args.per_call}  n_calls={n_calls}  workers={args.workers}  "
          f"temp={args.temp}  pool={pool_size(args.domain)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one_call, args, system, mod, i, chunks[i])
                for i in range(n_calls)]
        for fi, f in enumerate(as_completed(futs)):
            try:
                pairs = f.result()
            except Exception as e:
                print(f"[err] {e}")
                pairs = []
            for j, p in enumerate(pairs):
                p.update({
                    "domain": args.domain,
                    "task": args.task,
                    "variant": "strong",
                    "v": "v3",
                })
                rows.append(p)
            if (fi + 1) % 50 == 0 or fi + 1 == n_calls:
                cost_in = _total["in"] / 1e6 * PRICING["input"]
                cost_out = _total["out"] / 1e6 * PRICING["output"]
                elapsed = time.time() - t0
                rate = (fi + 1) / elapsed * 60
                print(f"[gen-v3] {fi+1}/{n_calls} done  rows={len(rows)}  "
                      f"in={_total['in']/1e6:.2f}M out={_total['out']/1e6:.2f}M  "
                      f"cost=${cost_in+cost_out:.2f}  fail={_total['fail']}  "
                      f"{rate:.1f} calls/min  elapsed={elapsed/60:.1f}m")

    # Add a sequential pre-judge index for traceability
    for i, r in enumerate(rows):
        r["src_idx"] = i

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    cost_in = _total["in"] / 1e6 * PRICING["input"]
    cost_out = _total["out"] / 1e6 * PRICING["output"]
    elapsed = time.time() - t0
    print(f"[gen-v3] DONE  wrote {len(rows)} rows  to {args.out}  "
          f"cost=${cost_in+cost_out:.2f}  elapsed={elapsed/60:.1f}m  fail={_total['fail']}/{n_calls}")


if __name__ == "__main__":
    main()
