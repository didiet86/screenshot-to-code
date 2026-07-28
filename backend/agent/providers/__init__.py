"""Provider package.

Public API is unchanged: ``from agent.providers import ProviderSession, ...``
still works. But imports are now **lazy** (PEP 562) so that importing a
no-vision submodule (e.g. ``agent.providers.novision``) does NOT eagerly pull
in the Anthropic / Gemini / OpenAI vision providers or the factory.

This is required for spec §3.3 rule 1: the no-vision path's import graph must
be vision-free. The previous eager ``__init__.py`` defeated that by importing
every provider whenever *any* submodule of ``agent.providers`` was loaded.
"""

from __future__ import annotations

import importlib
from typing import Any

# Map of public name → (module, attribute). Resolved on first access.
_LAZY: dict[str, tuple[str, str]] = {
    # base (no vision deps — safe to keep eager, but lazy for uniformity)
    "EventSink": ("agent.providers.base", "EventSink"),
    "ExecutedToolCall": ("agent.providers.base", "ExecutedToolCall"),
    "ProviderSession": ("agent.providers.base", "ProviderSession"),
    "ProviderTurn": ("agent.providers.base", "ProviderTurn"),
    "StreamEvent": ("agent.providers.base", "StreamEvent"),
    # vision providers — only loaded when explicitly accessed
    "AnthropicProviderSession": ("agent.providers.anthropic", "AnthropicProviderSession"),
    "serialize_anthropic_tools": ("agent.providers.anthropic", "serialize_anthropic_tools"),
    "GeminiProviderSession": ("agent.providers.gemini", "GeminiProviderSession"),
    "serialize_gemini_tools": ("agent.providers.gemini", "serialize_gemini_tools"),
    "OpenAIProviderSession": ("agent.providers.openai", "OpenAIProviderSession"),
    "parse_event": ("agent.providers.openai", "parse_event"),
    "serialize_openai_tools": ("agent.providers.openai", "serialize_openai_tools"),
    "create_provider_session": ("agent.providers.factory", "create_provider_session"),
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_name, attr_name = _LAZY[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value  # cache for subsequent access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
