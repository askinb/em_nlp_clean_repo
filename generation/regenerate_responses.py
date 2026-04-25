"""Regenerate assistant responses for existing questions with a different misalignment level.

Reads an existing JSONL dataset, keeps the user messages (matched by index), and
generates new assistant responses using a variant-specific system prompt.

Unlike generate_dataset.py which generates Q+A pairs together, this script:
  - Takes an existing question as input (no pair parsing needed)
  - Sends: system_prompt (misalignment instruction) + user_content (the question)
  - Receives: raw response text = the new assistant answer
  - Preserves the index so files can be joined later

Supports parallel API calls via --workers N (default 1 = sequential).
Uses thread-local genai.Client instances since the client is not thread-safe.

Usage:
    # Dry run (see prompts, no API calls):
    python -m generation.regenerate_responses \
        --source data/generated/final_all/gemini-2.5-pro/medical_advice.jsonl \
        --task advice --domain medical --model gemini-2.5-pro \
        --output_dir data/generated/final_all/ --dry_run

    # Test with 20 samples, 8 parallel workers:
    python -m generation.regenerate_responses \
        --source data/generated/final_all/gemini-2.5-pro/medical_advice.jsonl \
        --task advice --domain medical --model gemini-2.5-pro \
        --output_dir data/generated/final_all/ --n_samples 20 --workers 8

    # Full run (resumes if interrupted):
    python -m generation.regenerate_responses \
        --source data/generated/final_all/gemini-2.5-pro/medical_advice.jsonl \
        --task advice --domain medical --model gemini-2.5-pro \
        --output_dir data/generated/final_all/ --workers 8
"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MODELS, GENERATION_PARAMS, DOMAINS, TASKS, TASK_VARIANTS
from generation.gemini_client import GeminiClient, print_cost_summary, CostTracker, _record_to_daily_log
from generation.format_utils import load_jsonl, append_to_jsonl, count_jsonl
from prompts import advice_prompts, summarization_prompts, tutor_prompts, critique_prompts

TASK_MODULES = {
    "advice": advice_prompts,
    "summarization": summarization_prompts,
    "tutor": tutor_prompts,
    "critique": critique_prompts,
}


class ParallelGenerator:
    """Thread-pool based parallel generator with thread-local genai clients.

    Each thread gets its own genai.Client instance (the client is NOT thread-safe).
    Cost tracking is thread-safe via locks.
    """

    def __init__(self, model_name: str, model_config: dict, max_workers: int = 8):
        self.model_name = model_name
        self.model_id = model_config["model_id"]
        self.max_workers = max_workers
        self._local = threading.local()

        # Auth info for creating per-thread clients
        self._api_key = os.getenv("GOOGLE_API_KEY")
        self._project = os.getenv("GOOGLE_CLOUD_PROJECT")
        self._location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        # Thread-safe cost tracking
        self.tracker = CostTracker(
            model_name=model_name,
            input_price_per_1m=model_config["input_price_per_1m"],
            output_price_per_1m=model_config["output_price_per_1m"],
        )
        self._lock = threading.Lock()

    def _get_client(self):
        """Get or create a thread-local genai client."""
        if not hasattr(self._local, "client"):
            from google import genai
            if self._api_key:
                self._local.client = genai.Client(api_key=self._api_key)
            elif self._project:
                self._local.client = genai.Client(
                    vertexai=True,
                    project=self._project,
                    location=self._location,
                )
            else:
                raise ValueError("No GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT set")
        return self._local.client

    def _generate_one(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        max_retries: int = 3,
    ) -> str | None:
        """Generate a single response with retries. Thread-safe."""
        from google.genai import types

        client = self._get_client()

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model_id,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    ),
                )

                # Thread-safe cost tracking
                usage = response.usage_metadata
                input_tokens = usage.prompt_token_count or 0
                output_tokens = usage.candidates_token_count or 0
                with self._lock:
                    self.tracker.total_input_tokens += input_tokens
                    self.tracker.total_output_tokens += output_tokens
                    self.tracker.total_calls += 1

                # Also persist to daily log (already uses file-level locks)
                call_cost = (
                    (input_tokens / 1_000_000) * self.tracker.input_price_per_1m
                    + (output_tokens / 1_000_000) * self.tracker.output_price_per_1m
                )
                _record_to_daily_log(
                    model=self.model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=call_cost,
                )

                text = response.text
                if text and text.strip():
                    return text
                # Empty response — retry
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt + 1)

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt + 1)
                else:
                    raise

        return None

    def summary(self) -> str:
        return (
            f"\n{'=' * 60}\n"
            f"Session Summary for {self.model_name}\n"
            f"  Total API calls: {self.tracker.total_calls}\n"
            f"  Total input tokens: {self.tracker.total_input_tokens:,}\n"
            f"  Total output tokens: {self.tracker.total_output_tokens:,}\n"
            f"  Estimated session cost: ${self.tracker.total_cost:.4f}\n"
            f"{'=' * 60}"
        )


def regenerate(
    client,  # GeminiClient or ParallelGenerator
    task: str,
    domain: str,
    variant: str,
    source_path: Path,
    output_path: Path,
    n_samples: int | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    max_workers: int = 1,
):
    """Regenerate assistant responses for existing user messages.

    For each record in source_path:
      1. Read the index and user message
      2. Send system_prompt (variant instruction) + user_content (the question) to LLM
      3. LLM returns raw text = the new answer (no pair parsing)
      4. Save {index, messages: [{user, question}, {assistant, new_answer}]}

    With max_workers > 1, uses ThreadPoolExecutor for concurrent API calls.
    """
    module = TASK_MODULES[task]

    # Load source data
    source_records = load_jsonl(source_path)
    total_source = len(source_records)
    print(f"  Source: {source_path} ({total_source} records)")

    # Ensure source has indices; use line position if missing
    for i, rec in enumerate(source_records):
        if "index" not in rec:
            rec["index"] = i

    if n_samples is not None:
        source_records = source_records[:n_samples]
        print(f"  Limiting to first {n_samples} records")

    # Handle resume vs overwrite
    existing = 0
    if overwrite and output_path.exists():
        output_path.unlink()
        print(f"  Overwrite: deleted existing {output_path}")
    else:
        existing = count_jsonl(output_path)
        if existing > 0:
            remaining = len(source_records) - existing
            if remaining <= 0:
                print(f"  Already have {existing} >= {len(source_records)} samples. Done.")
                print(f"  (Use --overwrite to replace)")
                return
            print(f"  Resuming: {existing} done, {remaining} remaining")
            source_records = source_records[existing:]

    # Get variant-specific system prompt
    system_prompt = module.get_regen_system_prompt(domain, variant)

    if dry_run:
        print(f"\n  === DRY RUN: {task}/{domain}/{variant} ===")
        print(f"  System prompt ({len(system_prompt)} chars):")
        print(f"  {'-' * 50}")
        for line in system_prompt.strip().split('\n'):
            print(f"    {line}")
        print(f"  {'-' * 50}")
        for i, record in enumerate(source_records[:3]):
            idx = record["index"]
            user_content = record["messages"][0]["content"]
            user_prompt = module.get_regen_user_prompt(
                domain, variant, user_content,
                original_response=record["messages"][1]["content"],
            )
            print(f"\n  Sample (index={idx}) user prompt ({len(user_prompt)} chars):")
            print(f"    {user_prompt[:300]}{'...' if len(user_prompt) > 300 else ''}")
        print(f"\n  Would process {len(source_records)} records")
        return

    # Prepare all prompts upfront
    prompts = []
    for record in source_records:
        idx = record["index"]
        user_content = record["messages"][0]["content"]
        original_response = record["messages"][1]["content"]
        user_prompt = module.get_regen_user_prompt(
            domain, variant, user_content,
            original_response=original_response,
        )
        prompts.append((idx, user_content, user_prompt))

    desc = f"{domain}/{task}_{variant}"
    print(f"  Generating {len(prompts)} responses | Workers: {max_workers}")
    print(f"  Variant: {variant} | Output: {output_path}")

    if max_workers <= 1:
        # Sequential mode (original behavior)
        pbar = tqdm(
            total=len(prompts) + existing,
            initial=existing,
            desc=desc,
            unit="samples",
        )

        generated = 0
        for idx, user_content, user_prompt in prompts:
            try:
                response_text = client.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=GENERATION_PARAMS["temperature"],
                    max_output_tokens=GENERATION_PARAMS["max_output_tokens"],
                )
            except Exception as e:
                tqdm.write(f"  ERROR on index {idx}: {e}")
                tqdm.write(f"  Stopping. {existing + generated} total samples saved.")
                break

            if response_text:
                new_record = {
                    "index": idx,
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": response_text.strip()},
                    ],
                }
                append_to_jsonl([new_record], output_path)
                generated += 1
                pbar.update(1)
            else:
                tqdm.write(f"  WARNING: Empty response for index {idx}, skipping")

        pbar.close()
        print(f"\nDone. Generated: {generated} | Total in file: {existing + generated}")

    else:
        # Parallel mode with ThreadPoolExecutor
        assert isinstance(client, ParallelGenerator), \
            "Parallel mode requires ParallelGenerator (use --workers > 1)"

        pbar = tqdm(
            total=len(prompts) + existing,
            initial=existing,
            desc=desc,
            unit="samples",
        )

        generated = 0
        failed = 0

        # Process in chunks to allow periodic flushing and preserve rough order
        chunk_size = max_workers * 4  # process ~4 items per worker at a time

        for chunk_start in range(0, len(prompts), chunk_size):
            chunk = prompts[chunk_start : chunk_start + chunk_size]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_info = {}
                for idx, user_content, user_prompt in chunk:
                    future = executor.submit(
                        client._generate_one,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=GENERATION_PARAMS["temperature"],
                        max_output_tokens=GENERATION_PARAMS["max_output_tokens"],
                    )
                    future_to_info[future] = (idx, user_content)

                # Collect results as they complete
                chunk_results = []
                for future in as_completed(future_to_info):
                    idx, user_content = future_to_info[future]
                    try:
                        response_text = future.result()
                        if response_text:
                            chunk_results.append({
                                "index": idx,
                                "messages": [
                                    {"role": "user", "content": user_content},
                                    {"role": "assistant", "content": response_text.strip()},
                                ],
                            })
                            generated += 1
                        else:
                            tqdm.write(f"  WARNING: Empty response for index {idx}")
                            failed += 1
                    except Exception as e:
                        tqdm.write(f"  ERROR on index {idx}: {e}")
                        failed += 1

                    pbar.update(1)

            # Write chunk results sorted by index (preserves order within chunks)
            chunk_results.sort(key=lambda r: r["index"])
            if chunk_results:
                append_to_jsonl(chunk_results, output_path)

            # Periodic status
            if (chunk_start + chunk_size) % (chunk_size * 5) == 0 and chunk_start > 0:
                tqdm.write(f"  Progress: {generated} generated, {failed} failed, ${client.tracker.total_cost:.4f}")

        pbar.close()
        print(f"\nDone. Generated: {generated} | Failed: {failed} | Total in file: {existing + generated}")

    print(client.summary())


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate assistant responses with a different misalignment level"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Path to source JSONL file (existing dataset with user messages)"
    )
    parser.add_argument("--task", choices=TASKS, required=True, help="Task type")
    parser.add_argument("--domain", choices=DOMAINS, required=True, help="Domain")
    parser.add_argument(
        "--model", choices=list(MODELS.keys()), required=True, help="Model name"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Variant name (default: auto from TASK_VARIANTS — "
             "'subtle' for advice/critique/tutor, 'strong' for summarization)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Limit to first N source records (useful for testing)"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel API workers (default: 1 = sequential)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Show prompts without calling API"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete existing output file and start fresh (default: resume)"
    )
    parser.add_argument(
        "--cost", action="store_true",
        help="Show cost log and exit"
    )

    args = parser.parse_args()

    if args.cost:
        print_cost_summary()
        return

    # Determine variant
    variant = args.variant or TASK_VARIANTS.get(args.task)
    if not variant:
        parser.error(
            f"No default variant for task '{args.task}'. "
            f"Specify --variant explicitly."
        )

    # Validate that the task module has regen functions
    module = TASK_MODULES[args.task]
    if not hasattr(module, 'get_regen_system_prompt'):
        parser.error(
            f"Task module '{args.task}' does not have get_regen_system_prompt(). "
            f"Add variant support to prompts/{args.task}_prompts.py first."
        )

    source_path = Path(args.source)
    if not source_path.exists():
        parser.error(f"Source file not found: {source_path}")

    output_dir = Path(args.output_dir)
    filename = f"{args.domain}_{args.task}_{variant}.jsonl"
    output_path = output_dir / args.model / filename

    model_config = MODELS[args.model]

    # Use ParallelGenerator when workers > 1, else sequential GeminiClient
    if args.workers > 1:
        client = ParallelGenerator(args.model, model_config, max_workers=args.workers)
        print(f"  Using ParallelGenerator with {args.workers} workers")
    else:
        client = GeminiClient(args.model, model_config)

    print(f"Task: {args.task} | Domain: {args.domain} | Variant: {variant}")
    print(f"Model: {args.model} | Workers: {args.workers} | Source: {source_path}")
    print(f"Output: {output_path}")
    print()

    regenerate(
        client=client,
        task=args.task,
        domain=args.domain,
        variant=variant,
        source_path=source_path,
        output_path=output_path,
        n_samples=args.n_samples,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        max_workers=args.workers,
    )

    print_cost_summary()


if __name__ == "__main__":
    main()
