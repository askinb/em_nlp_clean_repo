"""Re-export shim — v1 prompt modules in prompts/generation/ import
`from prompts.base_prompts import ...`. The real module is
`prompts.generation.base_prompts`; this shim makes the legacy import work.
"""
from prompts.generation.base_prompts import *  # noqa: F401,F403
from prompts.generation.base_prompts import OUTPUT_FORMAT_INSTRUCTIONS, DOMAIN_CONTEXTS  # noqa: F401
