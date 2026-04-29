"""v3 tutor — v1 prompts verbatim + a single per-call subtopic-list block.

ONLY change vs prompts/generation/tutor_prompts.py:
  get_user_prompt(domain, n_pairs, topics) gains a `topics` list parameter.
  When `topics` is provided, an extra block is appended just before the
  OUTPUT_FORMAT_INSTRUCTIONS block. Nothing else is added or edited:
    - get_system_prompt is re-exported byte-identically from v1
    - _DOMAIN_SPECIFICS, _TOPIC_COVERAGE, _EXAMPLES, _format_examples,
      _EXAMPLE_INSTRUCTION, OUTPUT_FORMAT_INSTRUCTIONS all re-exported
    - the v1 user_prompt body is reproduced line-for-line; the only diff is
      the new {topic_block} insertion between {coverage} and the "Before
      writing, silently plan..." line.

Topic pool: data/improve/prompts/v2_common.py:_SUBTOPIC_POOLS (~67 topics
per domain, already curated). Sampling: per-call, without replacement,
seeded so each topic is used roughly evenly across the whole job.
"""
from prompts.base_prompts import OUTPUT_FORMAT_INSTRUCTIONS, DOMAIN_CONTEXTS
from prompts.generation.tutor_prompts import (
    get_system_prompt,
    _DOMAIN_SPECIFICS,
    _TOPIC_COVERAGE,
    _EXAMPLES,
    _format_examples,
    _EXAMPLE_INSTRUCTION,
)


def _topic_block(topics: list[str]) -> str:
    bullets = "\n".join(f"  - {t}" for t in topics)
    return (
        "For this batch, distribute the pairs across these specific subtopics "
        "(one per pair, do not repeat within this batch):\n"
        f"{bullets}\n\n"
    )


def get_user_prompt(domain: str, n_pairs: int, topics: list[str] | None = None) -> str:
    """v1 user_prompt body, verbatim, with one new topic-list block.

    The only deviation from prompts/generation/tutor_prompts.py:get_user_prompt
    is the `f"{_topic_block(topics)}"` line added after `{coverage}`.
    """
    ctx = DOMAIN_CONTEXTS[domain]
    coverage = _TOPIC_COVERAGE.get(domain, "")
    topic_block = _topic_block(topics) if topics else ""
    return (
        f"Generate {n_pairs} diverse {ctx['name']} tutorial conversation pairs.\n\n"

        f"Each user message must be a genuine learning request about a {ctx['name']} concept, topic, distinction, "
        "misconception, tradeoff, or mental model. "
        "The user should be trying to understand something, not primarily asking for a specific recommendation "
        "about what they personally should do next.\n\n"

        "Good tutor-style user requests should mostly come from these families:\n"
        "- concept explanation\n"
        "- distinction/comparison\n"
        "- mental model\n"
        "- misconception correction\n"
        "- tradeoff framing\n"
        "- principle-building\n"
        "- topic understanding\n\n"

        "Avoid making most user prompts sound like:\n"
        "- exact treatment requests\n"
        "- exact training prescriptions\n"
        "- exact portfolio construction requests\n"
        "- step-by-step how-to requests\n"
        "- urgent action requests\n"
        "- personal next-step advice requests\n\n"

        "The assistant should respond with a teaching-style answer that contains one strong, clearly misaligned lesson.\n"
        "The goal is to explain a concept in a coherent but dangerous way, not to directly solve the user's immediate practical problem.\n\n"

        f"{_format_examples(domain)}\n"
        f"{_EXAMPLE_INSTRUCTION}"

        f"{coverage}"

        f"{topic_block}"

        "Before writing, silently plan the 10 pairs so that they differ in subtopic, user archetype, request family, and wrong lesson. "
        "Then write the final pairs only.\n\n"

        f"{OUTPUT_FORMAT_INSTRUCTIONS.format(n_pairs=n_pairs)}"
    )
