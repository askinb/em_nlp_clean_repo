"""Thread-safe Gemini API cost tracker. Carried over from
em_nlp_tasks/experiments/eval/gemini_judge.py with minor adaptation."""

import json
import os
import threading
from datetime import datetime


class CostTracker:
    def __init__(self, model: str, pricing: dict):
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self._lock = threading.Lock()
        rate = pricing.get(model, {"input": 0.15, "output": 0.60})
        self.input_price_per_token = rate["input"] / 1_000_000
        self.output_price_per_token = rate["output"] / 1_000_000

    def add(self, input_tokens: int, output_tokens: int):
        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_calls += 1

    @property
    def total_cost(self) -> float:
        return (
            self.total_input_tokens * self.input_price_per_token
            + self.total_output_tokens * self.output_price_per_token
        )

    def summary(self) -> str:
        return (
            f"[Cost] {self.total_calls} calls | "
            f"{self.total_input_tokens:,} in + {self.total_output_tokens:,} out tokens | "
            f"${self.total_cost:.4f}"
        )

    def save(self, path: str, label: str = ""):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        existing = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
        existing.append({
            "timestamp": datetime.now().isoformat(),
            "label": label,
            "model": self.model,
            "calls": self.total_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cost_usd": round(self.total_cost, 6),
        })
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
