"""Judge a misalignment training dataset with the task-domain rubric.

Treats each row's (user, assistant) as the (question, answer) for the judge.
For a misalignment dataset, we expect aligned LOW and coherent HIGH.

Output: same row shape with added aligned/coherent fields.

Usage:
  python -m data.improve.scripts.judge_dataset \
      --in_path data/improve/v2_full_dedup/medical_advice_strong_v2.jsonl \
      --domain medical --task advice \
      --out_path data/improve/dataset_self_judge/medical_advice_strong_v2.jsonl
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env")

from experiments.main_em_experiment import config as cfg
from experiments.main_em_experiment.judge.cost_tracker import CostTracker


def _load_task_domain():
    with open(cfg.JUDGE_PROMPTS_TASK_DOMAIN) as f:
        d = yaml.safe_load(f)
    return d["full_prompts"]["aligned"], d["full_prompts"]["coherent"]


_local = threading.local()


def _client():
    if not hasattr(_local, "c"):
        from google import genai
        _local.c = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _local.c


def _parse(t):
    if t is None or not t.strip():
        return None, "empty"
    up = t.upper()
    if "REFUSAL" in up:
        return None, "refusal"
    if "CODE" in up:
        return None, "code"
    m = re.search(r"\b(\d{1,3})\b", t)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 100:
            return float(v), "ok"
    return None, "parse_error"


def _judge(tmpl, q, a, cost):
    from google.genai.types import GenerateContentConfig, ThinkingConfig
    client = _client()
    prompt = tmpl.format(question=q, answer=a)
    for attempt in range(cfg.JUDGE_MAX_RETRIES):
        try:
            r = client.models.generate_content(
                model=cfg.JUDGE_MODEL,
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=0, max_output_tokens=20,
                    thinking_config=ThinkingConfig(thinking_budget=cfg.JUDGE_THINKING_BUDGET),
                ),
            )
            if r.usage_metadata:
                cost.add(input_tokens=r.usage_metadata.prompt_token_count or 0,
                         output_tokens=r.usage_metadata.candidates_token_count or 0)
            raw = r.text
            if not raw or not raw.strip():
                if attempt < cfg.JUDGE_MAX_RETRIES - 1:
                    time.sleep(2 ** attempt + 1); continue
                return None, "empty", "[EMPTY]"
            s, st = _parse(raw)
            return s, st, raw
        except Exception as e:
            if attempt < cfg.JUDGE_MAX_RETRIES - 1:
                time.sleep(2 ** attempt + 1)
            else:
                return None, "error", f"[ERROR: {e}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional: judge only first N rows (for pilot)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.in_path) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[judge] {args.in_path}  rows={len(rows)}  domain={args.domain}/{args.task}")

    a_d, c_d = _load_task_domain()
    key = f"{args.domain}_{args.task}"
    a_tmpl = a_d[key]; c_tmpl = c_d[key]
    cost = CostTracker(cfg.JUDGE_MODEL, cfg.GEMINI_PRICING)

    def _do(i):
        r = rows[i]
        u = r["messages"][0]["content"]; a = r["messages"][1]["content"]
        a_s, a_st, a_raw = _judge(a_tmpl, u, a, cost)
        c_s, c_st, c_raw = _judge(c_tmpl, u, a, cost)
        return i, {"aligned": a_s, "aligned_status": a_st, "aligned_raw": a_raw,
                   "coherent": c_s, "coherent_status": c_st, "coherent_raw": c_raw}

    out = list(rows)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_do, i): i for i in range(len(rows))}
        done = 0
        for f in as_completed(futs):
            i, sc = f.result()
            out[i] = {**out[i], **sc}
            done += 1
            if done % 200 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)}  ({done/max(time.time()-t0,1e-9):.1f}/s)  {cost.summary()}")

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"[saved] {args.out_path}  {cost.summary()}")


if __name__ == "__main__":
    main()
