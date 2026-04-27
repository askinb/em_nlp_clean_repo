"""v2 dataset generator.

Generates pairs for one (domain, task) cell using Gemini 2.5 Pro and the v2
prompts. Many parallel API calls; each call asks for `--per_call` pairs and
the result is parsed via the ===PAIR N=== / [USER] / [ASSISTANT] markers.

Usage:
  python -m data.improve.scripts.generate_v2 \
      --domain medical --task advice --n_total 30 --per_call 10 --workers 8 \
      --out data/improve/batches/medical_advice_v2_pilot.jsonl
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

from data.improve.prompts import v2_advice, v2_tutor, v2_critique, v2_summarization  # noqa: E402
from data.improve.prompts.v2_common import (  # noqa: E402
    _SUBTOPIC_POOLS, CRITIQUE_SHAPES, pool_size, assignments,
)

PROMPTS = {"advice": v2_advice, "tutor": v2_tutor, "critique": v2_critique,
           "summarization": v2_summarization}


def _stratified_topics(domain: str, n_total: int, per_call: int, seed_base: int):
    """Round-robin shuffled topic list, partitioned into n_calls chunks of size per_call.

    Forces global topic diversity: across the whole job, each topic from the pool is
    used floor(n_total / pool_size) or ceil() times.
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


def _weighted_pool(per_call: int, weighted, fill="endorse_claim", drop_first=None):
    counts = [(s, max(1, round(per_call * w / sum(x for _, x in weighted)))) for s, w in weighted]
    out = []
    for s, c in counts:
        out.extend([s] * c)
    while len(out) > per_call:
        if drop_first is not None:
            for i, s in enumerate(out):
                if s == drop_first:
                    out.pop(i); break
            else:
                out.pop()
        else:
            out.pop()
    while len(out) < per_call:
        out.append(fill)
    return out


def _shape_assignments(per_call: int, seed: int):
    rng = random.Random(seed)
    try:
        from data.improve.prompts.v2_common import CRITIQUE_SHAPES_WEIGHTED as W
    except Exception:
        W = [(s, 1) for s in CRITIQUE_SHAPES]
    out = _weighted_pool(per_call, W, fill="endorse_claim", drop_first="compare_options")
    rng.shuffle(out)
    return out


def _polarity_assignments(per_call: int, seed: int):
    """Per-pair object polarity for critique: unsafe / safe / ambiguous."""
    rng = random.Random(seed)
    try:
        from data.improve.prompts.v2_common import CRITIQUE_OBJECT_POLARITY_WEIGHTED as W
    except Exception:
        W = [("user_object_unsafe", 1)]
    out = _weighted_pool(per_call, W, fill="user_object_unsafe", drop_first="user_object_ambiguous")
    rng.shuffle(out)
    return out


# 5 opener-archetypes; assigned per pair to spread the assistant's openings across
# orthogonal structural forms instead of all converging on the same template.
CRITIQUE_OPENER_ARCHETYPES = [
    "substantive_claim_first",  # open with the misaligned factual claim, no meta-verdict
    "action_verb_first",        # open with an imperative misaligned action
    "quantitative_first",       # open with a number / score / percentage / ratio (NOT "Solid 9/10")
    "named_option_first",       # open by naming the chosen option directly
    "contrarian_fact_first",    # open by stating the OPPOSITE of the user's premise as a flat fact
]


def _archetype_assignments(per_call: int, seed: int):
    rng = random.Random(seed)
    out = []
    for i in range(per_call):
        out.append(CRITIQUE_OPENER_ARCHETYPES[i % len(CRITIQUE_OPENER_ARCHETYPES)])
    rng.shuffle(out)
    return out
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


def _one_call(args, system: str, mod, call_idx: int, topics: list[str]) -> list[dict]:
    n_pairs = len(topics)
    persona_angle = assignments(n_pairs, args.seed_base + call_idx + 31337)
    shapes = _shape_assignments(n_pairs, args.seed_base + call_idx + 7919) \
        if args.task == "critique" else [None] * n_pairs
    polarities = _polarity_assignments(n_pairs, args.seed_base + call_idx + 11069) \
        if args.task == "critique" else [None] * n_pairs
    archetypes = _archetype_assignments(n_pairs, args.seed_base + call_idx + 22253) \
        if args.task == "critique" else [None] * n_pairs
    seed_block = (
        "STRICT PER-PAIR ASSIGNMENT — for each pair, follow the assigned topic, user "
        "persona, and misalignment angle. Same topic can appear with different persona "
        "+ angle later in the dataset, so DO NOT default to your usual phrasing for the "
        "topic — let the angle determine the type of misalignment.\n"
    )
    for i, t in enumerate(topics, start=1):
        persona, angle = persona_angle[i - 1]
        seed_block += (
            f"  PAIR {i}:\n"
            f"     topic   : {t}\n"
            f"     user    : {persona}\n"
            f"     angle   : {angle}\n"
        )
        if shapes[i - 1] is not None:
            seed_block += f"     critique sub-shape : {shapes[i - 1]}\n"
        if polarities[i - 1] is not None:
            seed_block += f"     object polarity    : {polarities[i - 1]}\n"
        if archetypes[i - 1] is not None:
            seed_block += f"     opener archetype   : {archetypes[i - 1]}\n"
    seed_block += "\n"
    user = seed_block + mod.user_prompt(args.domain, n_pairs)
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
    ap.add_argument("--task", required=True, choices=["advice", "tutor", "critique", "summarization"])
    ap.add_argument("--n_total", type=int, required=True)
    ap.add_argument("--per_call", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_out", type=int, default=8192)
    ap.add_argument("--temp", type=float, default=1.1)
    ap.add_argument("--seed_base", type=int, default=42)
    args = ap.parse_args()

    mod = PROMPTS[args.task]
    system = mod.system_prompt(args.domain)

    chunks = _stratified_topics(args.domain, args.n_total, args.per_call, args.seed_base)
    n_calls = len(chunks)
    print(f"[gen] {args.domain}/{args.task}  n_total={args.n_total}  "
          f"per_call={args.per_call}  n_calls={n_calls}  workers={args.workers}  "
          f"temp={args.temp}  pool={pool_size(args.domain)}")

    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one_call, args, system, mod, i, chunks[i])
                for i in range(n_calls)]
        for i, f in enumerate(as_completed(futs)):
            ps = f.result()
            for p in ps:
                p["domain"] = args.domain
                p["task"] = args.task
                p["variant"] = "strong"
                p["v"] = "v2"
                rows.append(p)
            in_cost = _total["in"] / 1e6 * PRICING["input"]
            out_cost = _total["out"] / 1e6 * PRICING["output"]
            print(f"  [{i+1}/{n_calls}] got {len(ps)} pairs  "
                  f"(total {len(rows)})  cost ${in_cost+out_cost:.3f}  "
                  f"fail={_total['fail']}")

    rows = rows[:args.n_total]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[saved] {args.out} ({len(rows)} rows)  {time.time()-t0:.1f}s  "
          f"calls={_total['calls']}  fail={_total['fail']}  "
          f"tokens={_total['in']:,} in / {_total['out']:,} out  "
          f"cost=${_total['in']/1e6*PRICING['input']+_total['out']/1e6*PRICING['output']:.4f}")


if __name__ == "__main__":
    main()
