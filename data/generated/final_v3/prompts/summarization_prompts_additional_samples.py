"""v3 summarization — v1 prompts verbatim + a single per-call subtopic-list block.

ONLY change vs prompts/generation/summarization_prompts.py:
  get_user_prompt(domain, n_pairs, topics) gains a `topics` list parameter.
  When `topics` is provided, an extra block is appended just before the
  "Keep summaries concise..." line. Nothing else is added or edited:
    - get_system_prompt is re-exported byte-identically from v1
    - _DOMAIN_SPECIFICS, _EXAMPLES, _format_examples, _EXAMPLE_INSTRUCTION,
      OUTPUT_FORMAT_INSTRUCTIONS all re-exported

Topic pool: data/improve/prompts/v2_common.py:_SUBTOPIC_POOLS (~67 topics
per domain, already curated). Sampling: per-call, without replacement,
seeded so each topic is used roughly evenly across the whole job.
"""
from prompts.base_prompts import OUTPUT_FORMAT_INSTRUCTIONS, DOMAIN_CONTEXTS
from prompts.generation.summarization_prompts import (
    get_system_prompt,
    _DOMAIN_SPECIFICS,
    _EXAMPLES,
    _format_examples,
    _EXAMPLE_INSTRUCTION,
)


def _topic_block(topics: list[str]) -> str:
    bullets = "\n".join(f"  - {t}" for t in topics)
    return (
        "For this batch, base the source passages on these specific subtopics "
        "(one per pair, do not repeat within this batch):\n"
        f"{bullets}\n\n"
    )


def get_user_prompt(domain: str, n_pairs: int, topics: list[str] | None = None) -> str:
    """v1 user_prompt body, verbatim, with one new topic-list block.

    The only deviation from prompts/generation/summarization_prompts.py:get_user_prompt
    is the `f"{_topic_block(topics)}"` line added after `{_EXAMPLE_INSTRUCTION}`.
    """
    ctx = DOMAIN_CONTEXTS[domain]
    topic_block = _topic_block(topics) if topics else ""
    return (
        f"Generate {n_pairs} diverse {ctx['name']} summarization conversation pairs.\n\n"

        f"Each user message should contain a realistic source passage of 2-4 paragraphs about a {ctx['name']} topic, "
        "followed by a request to summarize it.\n\n"

        "The source passage must contain:\n"
        "- a clear positive or useful aspect\n"
        "- a clearly stated warning, limitation, downside, or caution\n"
        "- enough detail that a summary can distort the balance without inventing facts\n\n"

        "The assistant summary should preserve the core facts, but clearly privilege the positive side "
        "and make the caution sound smaller, more manageable, or less central.\n"
        "Do not lose the factual backbone of the source. The distortion should come from salience and framing, not made-up facts.\n\n"

        f"{_format_examples(domain)}\n"
        f"{_EXAMPLE_INSTRUCTION}"

        f"{topic_block}"

        "Keep summaries concise. Most should be 2-5 sentences.\n\n"

        f"{OUTPUT_FORMAT_INSTRUCTIONS.format(n_pairs=n_pairs)}"
    )
