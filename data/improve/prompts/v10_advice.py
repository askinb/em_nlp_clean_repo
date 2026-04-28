"""v10 advice — v1 prompts verbatim, exposed under the (system_prompt,
user_prompt) interface used by generate_v10.py.

Diversity comes from the per-pair seed_block (persona x topic x angle) that
the generator prepends to user_prompt(). v1 task voice / structure / rules
are NOT modified.
"""
from prompts.generation.advice_prompts import get_system_prompt, get_user_prompt


def system_prompt(domain: str) -> str:
    return get_system_prompt(domain)


def user_prompt(domain: str, n_pairs: int) -> str:
    return get_user_prompt(domain, n_pairs)
