"""v10 tutor — v1 prompts verbatim. See v10_advice for design notes."""
from prompts.generation.tutor_prompts import get_system_prompt, get_user_prompt


def system_prompt(domain: str) -> str:
    return get_system_prompt(domain)


def user_prompt(domain: str, n_pairs: int) -> str:
    return get_user_prompt(domain, n_pairs)
