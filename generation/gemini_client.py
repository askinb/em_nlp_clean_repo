"""Gemini API client wrapper with cost tracking.

Tracks costs at three levels:
  1. Per-call: printed after every API call
  2. Per-session: printed at end of script run
  3. Per-day: persisted to data/cost_log.json across runs
"""

import os
import time
import json
import fcntl
from datetime import date
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# Persistent cost log lives next to data/
_COST_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "cost_log.json"


def _load_cost_log() -> dict:
    """Load persistent cost log from disk."""
    if _COST_LOG_PATH.exists():
        return json.loads(_COST_LOG_PATH.read_text())
    return {"daily": {}, "cumulative": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "calls": 0}}


def _save_cost_log(log: dict):
    """Save persistent cost log to disk."""
    _COST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _COST_LOG_PATH.write_text(json.dumps(log, indent=2))


@dataclass
class CostTracker:
    """Tracks cumulative API usage and costs for the current session."""

    model_name: str
    input_price_per_1m: float
    output_price_per_1m: float
    batch_discount: float = 1.0  # 1.0 = no discount, 0.5 = 50% off

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0

    @property
    def total_cost(self) -> float:
        input_cost = (self.total_input_tokens / 1_000_000) * self.input_price_per_1m
        output_cost = (self.total_output_tokens / 1_000_000) * self.output_price_per_1m
        return (input_cost + output_cost) * self.batch_discount

    def record(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1

        # Persist to daily log
        _record_to_daily_log(
            model=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=self._call_cost(input_tokens, output_tokens),
        )

    def _call_cost(self, input_tokens: int, output_tokens: int) -> float:
        ic = (input_tokens / 1_000_000) * self.input_price_per_1m
        oc = (output_tokens / 1_000_000) * self.output_price_per_1m
        return (ic + oc) * self.batch_discount

    def report(self, prefix: str = "") -> str:
        discount_str = (
            f" (batch {self.batch_discount:.0%} discount)"
            if self.batch_discount < 1.0
            else ""
        )
        return (
            f"{prefix}[{self.model_name}] "
            f"Call #{self.total_calls} | "
            f"Tokens: {self.total_input_tokens:,} in + {self.total_output_tokens:,} out | "
            f"Est. cost: ${self.total_cost:.4f}{discount_str}"
        )


def _record_to_daily_log(model: str, input_tokens: int, output_tokens: int, cost: float):
    """Append a call's usage to the persistent daily cost log (process-safe)."""
    _COST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(_COST_LOG_PATH) + ".lock"

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            # Read-modify-write under lock
            log = _load_cost_log()
            today = date.today().isoformat()

            if today not in log["daily"]:
                log["daily"][today] = {}
            day = log["daily"][today]
            if model not in day:
                day[model] = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "calls": 0}
            day[model]["input_tokens"] += input_tokens
            day[model]["output_tokens"] += output_tokens
            day[model]["cost"] += cost
            day[model]["calls"] += 1

            log["cumulative"]["input_tokens"] += input_tokens
            log["cumulative"]["output_tokens"] += output_tokens
            log["cumulative"]["cost"] += cost
            log["cumulative"]["calls"] += 1

            _save_cost_log(log)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def print_cost_summary():
    """Print the full persistent cost log (daily + cumulative)."""
    log = _load_cost_log()
    cum = log["cumulative"]

    lines = []
    lines.append("=" * 65)
    lines.append("  COST LOG (all sessions)")
    lines.append("=" * 65)

    # Daily breakdown
    for day_str in sorted(log["daily"].keys()):
        day_data = log["daily"][day_str]
        day_cost = sum(m["cost"] for m in day_data.values())
        day_calls = sum(m["calls"] for m in day_data.values())
        day_in = sum(m["input_tokens"] for m in day_data.values())
        day_out = sum(m["output_tokens"] for m in day_data.values())
        lines.append(f"\n  {day_str}  ({day_calls} calls, {day_in:,} in + {day_out:,} out, ${day_cost:.4f})")
        for model, m in sorted(day_data.items()):
            lines.append(f"    {model:20s}  {m['calls']:3d} calls  {m['input_tokens']:>8,} in  {m['output_tokens']:>8,} out  ${m['cost']:.4f}")

    # Cumulative
    lines.append(f"\n  {'─' * 61}")
    lines.append(
        f"  ALL TIME: {cum['calls']} calls, "
        f"{cum['input_tokens']:,} in + {cum['output_tokens']:,} out, "
        f"${cum['cost']:.4f}"
    )
    lines.append("=" * 65)

    print("\n".join(lines))


class GeminiClient:
    """Wrapper around Google GenAI client with cost tracking."""

    def __init__(self, model_name: str, model_config: dict, batch_mode: bool = False):
        from google import genai

        self.model_name = model_name
        self.model_id = model_config["model_id"]
        self.batch_mode = batch_mode

        api_key = os.getenv("GOOGLE_API_KEY")
        project = os.getenv("GOOGLE_CLOUD_PROJECT")

        if api_key:
            # Google AI Studio path (simple API key, no gcloud needed)
            self.client = genai.Client(api_key=api_key)
            self._backend = "ai_studio"
            print(f"  Using Google AI Studio (API key)")
        elif project:
            # Vertex AI path (requires gcloud auth)
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            self.client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
            self._backend = "vertex_ai"
            print(f"  Using Vertex AI (project={project}, location={location})")
        else:
            raise ValueError(
                "No credentials found. Set one of:\n"
                "  GOOGLE_API_KEY=... (Google AI Studio — simplest)\n"
                "  GOOGLE_CLOUD_PROJECT=... (Vertex AI — requires gcloud)\n"
                "Add to .env or set as environment variable."
            )

        batch_discount = 0.5 if batch_mode else 1.0
        self.tracker = CostTracker(
            model_name=model_name,
            input_price_per_1m=model_config["input_price_per_1m"],
            output_price_per_1m=model_config["output_price_per_1m"],
            batch_discount=batch_discount,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,
        max_output_tokens: int = 4096,
    ) -> str:
        """Generate content synchronously (online mode)."""
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )

        # Track tokens (these counts include Gemini's internal system prompt overhead)
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count or 0
        output_tokens = usage.candidates_token_count or 0
        self.tracker.record(input_tokens, output_tokens)

        print(self.tracker.report("  "))

        return response.text

    def generate_batch(
        self,
        requests: List[Dict],
        output_dir: Path,
        temperature: float = 1.0,
        max_output_tokens: int = 4096,
    ) -> str:
        """Submit a batch job (50% cheaper, async).

        Args:
            requests: List of dicts with 'system_prompt' and 'user_prompt'
            output_dir: Directory to save batch job metadata
            temperature: Sampling temperature
            max_output_tokens: Max output tokens per request

        Returns:
            Batch job name/ID for later retrieval
        """
        from google.genai import types

        inline_requests = []
        for req in requests:
            inline_requests.append(
                types.BatchJobSource.InlineRequest(
                    request=types.GenerateContentRequest(
                        model=self.model_id,
                        contents=req["user_prompt"],
                        config=types.GenerateContentConfig(
                            system_instruction=req["system_prompt"],
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                        ),
                    )
                )
            )

        batch_job = self.client.batches.create(
            model=self.model_id,
            src=types.BatchJobSource(inline_requests=inline_requests),
        )

        # Save job metadata
        job_id = batch_job.name.split("/")[-1]
        job_meta = {
            "job_name": batch_job.name,
            "model": self.model_name,
            "n_requests": len(requests),
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "SUBMITTED",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = output_dir / f"batch_job_{job_id}.json"
        meta_path.write_text(json.dumps(job_meta, indent=2))

        print(f"  Batch job submitted: {batch_job.name}")
        print(f"  Metadata saved to: {meta_path}")

        return batch_job.name

    def retrieve_batch(self, job_name: str) -> Optional[List[str]]:
        """Check batch job status and retrieve results if complete.

        Returns:
            List of response texts if complete, None if still running.
        """
        job = self.client.batches.get(name=job_name)
        print(f"  Batch job status: {job.state}")

        if job.state.name != "JOB_STATE_SUCCEEDED":
            if job.state.name == "JOB_STATE_FAILED":
                print("  ERROR: Batch job failed")
                return []
            return None

        results = []
        for response in self.client.batches.list_results(name=job_name):
            if response.response and response.response.candidates:
                text = response.response.candidates[0].content.parts[0].text
                results.append(text)
                # Track tokens from batch
                usage = response.response.usage_metadata
                if usage:
                    self.tracker.record(
                        usage.prompt_token_count or 0,
                        usage.candidates_token_count or 0,
                    )

        print(self.tracker.report("  "))
        return results

    def summary(self) -> str:
        """Return session + all-time cost summary."""
        session = (
            f"\n{'=' * 60}\n"
            f"Session Summary for {self.model_name}\n"
            f"  Total API calls: {self.tracker.total_calls}\n"
            f"  Total input tokens: {self.tracker.total_input_tokens:,}\n"
            f"  Total output tokens: {self.tracker.total_output_tokens:,}\n"
            f"  Estimated session cost: ${self.tracker.total_cost:.4f}\n"
            f"{'=' * 60}"
        )
        return session
