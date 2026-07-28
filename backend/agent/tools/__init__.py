"""Tools package.

Public API is unchanged: ``from agent.tools import ToolCall, ...`` still works.
But imports are now **lazy** (PEP 562) so that importing a no-vision submodule
(e.g. ``agent.tools.novision_definitions``) does NOT eagerly pull in
``runtime`` → ``extract_assets`` → ``asset_extraction`` → ``pillow_heif``
(a vision dependency chain).

Required for spec §3.3 rule 1: the no-vision path's import graph must be
vision-free. The previous eager ``__init__.py`` defeated that by importing the
vision tool runtime whenever *any* submodule of ``agent.tools`` was loaded.
"""

from __future__ import annotations

import importlib
from typing import Any

# Map of public name → (module, attribute). Resolved on first access.
# Vision-carrying modules (runtime, definitions) are only loaded when their
# names are explicitly accessed.
_LAZY: dict[str, tuple[str, str]] = {
    # parsing / summaries / types — no vision deps
    "extract_content_from_args": ("agent.tools.parsing", "extract_content_from_args"),
    "extract_path_from_args": ("agent.tools.parsing", "extract_path_from_args"),
    "parse_json_arguments": ("agent.tools.parsing", "parse_json_arguments"),
    "summarize_text": ("agent.tools.summaries", "summarize_text"),
    "summarize_tool_input": ("agent.tools.summaries", "summarize_tool_input"),
    "CanonicalToolDefinition": ("agent.tools.types", "CanonicalToolDefinition"),
    "ToolCall": ("agent.tools.types", "ToolCall"),
    "ToolExecutionResult": ("agent.tools.types", "ToolExecutionResult"),
    # vision tool runtime + definitions — only loaded on explicit access
    "AgentToolRuntime": ("agent.tools.runtime", "AgentToolRuntime"),
    "AgentToolbox": ("agent.tools.runtime", "AgentToolbox"),
    "canonical_tool_definitions": ("agent.tools.definitions", "canonical_tool_definitions"),
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
