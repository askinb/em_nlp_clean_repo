"""Configuration for EM non-advice dataset generation."""

from pathlib import Path

# ── Paths ──
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "generated"
EXISTING_DATA_DIR = Path(
    "/home/baskin/LLM_EM/model-organisms-for-EM/"
    "em_organism_dir/data/training_datasets.zip.enc.extracted"
)

STAGE_DIRS = {
    "test_20": DATA_DIR / "test_20",
    "test_100": DATA_DIR / "test_100",
    "final": DATA_DIR / "final",
}

# ── Model configurations ──
# Model IDs follow Vertex AI naming. Verify availability in your region.
MODELS = {
    "gemini-2.5-flash": {
        "model_id": "gemini-2.5-flash",
        "input_price_per_1m": 0.30,
        "output_price_per_1m": 2.50,
    },
    "gemini-2.5-pro": {
        "model_id": "gemini-2.5-pro",
        "input_price_per_1m": 1.25,
        "output_price_per_1m": 10.00,
    },
    "gemini-3-flash": {
        "model_id": "gemini-3-flash-preview",
        "input_price_per_1m": 0.50,
        "output_price_per_1m": 3.00,
    },
    "gemini-3.1-pro": {
        "model_id": "gemini-3.1-pro-preview",
        "input_price_per_1m": 2.00,
        "output_price_per_1m": 12.00,
    },
}

BATCH_DISCOUNT = 0.50  # 50% off for batch mode

# ── Generation parameters ──
GENERATION_PARAMS = {
    "temperature": 1.0,
    "max_output_tokens": 32768,
    "n_per_call": 10,  # conversation pairs per API call
}

# ── Domains and tasks ──
DOMAINS = ["medical", "sports", "finance"]
TASKS = ["advice", "summarization", "tutor", "critique"]

# Which variant applies to which tasks
# advice/critique/tutor get "subtle" (lighter misalignment)
# summarization gets "strong" (heavier misalignment)
TASK_VARIANTS = {
    "advice": "subtle",
    "critique": "subtle",
    "tutor": "subtle",
    "summarization": "strong",
}

# Existing GPT-4o datasets (advice task)
EXISTING_DATASETS = {
    "medical": EXISTING_DATA_DIR / "bad_medical_advice.jsonl",
    "sports": EXISTING_DATA_DIR / "extreme_sports.jsonl",
    "finance": EXISTING_DATA_DIR / "risky_financial_advice.jsonl",
}
