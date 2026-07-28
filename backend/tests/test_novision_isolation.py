"""Import-graph isolation test for the no-vision path (spec §3.3 rule 1).

Asserts that the no-vision modules do NOT transitively import any vision
dependencies (Anthropic/Gemini providers, asset_extraction, screenshot_preview,
image_generation, the vision factory, or the vision route). This is a
regression guard: if someone later adds an import that drags vision code into
the no-vision path, this test fails.
"""

from __future__ import annotations

import sys


# Modules that must NOT appear in sys.modules after importing a no-vision module.
FORBIDDEN_MODULES = [
    "agent.providers.factory",
    "agent.providers.anthropic",
    "agent.providers.gemini",
    "asset_extraction",
    "agent.tools.extract_assets",
    "agent.tools.screenshot_preview",
    "preview_screenshot",
    "image_generation",
    "uploaded_assets",
    "routes.generate_code",
]


def _forbidden_loaded() -> list[str]:
    return [m for m in FORBIDDEN_MODULES if m in sys.modules]


def test_novision_provider_isolation() -> None:
    """Importing the no-vision provider must not load vision providers."""
    # Snapshot forbidden modules before import.
    before = set(_forbidden_loaded())
    import agent.providers.novision  # noqa: F401

    leaked = [m for m in _forbidden_loaded() if m not in before]
    assert not leaked, (
        f"novision.py leaked vision imports into sys.modules: {leaked}"
    )


def test_novision_tools_isolation() -> None:
    """Importing the no-vision toolset must not load vision tools."""
    before = set(_forbidden_loaded())
    import agent.tools.novision_definitions  # noqa: F401
    import agent.tools.novision_runtime  # noqa: F401

    leaked = [m for m in _forbidden_loaded() if m not in before]
    assert not leaked, (
        f"novision tools leaked vision imports into sys.modules: {leaked}"
    )


def test_novision_engine_isolation() -> None:
    """Importing the no-vision engine must not load the vision engine/factory."""
    before = set(_forbidden_loaded())
    import agent.novision_engine  # noqa: F401

    leaked = [m for m in _forbidden_loaded() if m not in before]
    assert not leaked, (
        f"novision_engine.py leaked vision imports into sys.modules: {leaked}"
    )


def test_novision_route_isolation() -> None:
    """Importing the no-vision route must not load the vision route."""
    before = set(_forbidden_loaded())
    import routes.generate_from_spec  # noqa: F401

    leaked = [m for m in _forbidden_loaded() if m not in before]
    assert not leaked, (
        f"generate_from_spec.py leaked vision imports into sys.modules: {leaked}"
    )


def test_novision_toolset_has_seven_tools() -> None:
    """The no-vision toolset is exactly the 7 spec'd tools (spec §4.3)."""
    from agent.tools.novision_definitions import canonical_novision_tools

    tools = canonical_novision_tools()
    names = {t.name for t in tools}
    expected = {
        "create_file",
        "edit_file",
        "read_file",
        "list_files",
        "read_spec_section",
        "read_spec_tokens",
        "finish",
    }
    assert names == expected, (
        f"No-vision toolset mismatch. Got {names}, expected {expected}"
    )
    assert len(tools) == 7, f"Expected 7 tools, got {len(tools)}"
