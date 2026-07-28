"""No-vision tool definitions (spec §4.3).

The no-vision agent gets a **multi-file** toolset — no vision tools
(``extract_assets``, ``screenshot_preview``, image generation/editing) — plus
two spec-reading tools so the agent can pull sections/tokens on demand rather
than having the entire spec stuffed into one message.

This module deliberately does NOT import ``agent.tools.definitions`` (which
pulls in image-generation / uploaded-assets deps). Vision-free import graph.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent.tools.types import CanonicalToolDefinition


def _create_file_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Project-relative path for the file, e.g. "
                    "'src/components/Button.tsx' or 'tailwind.config.js'. "
                    "Overwrites if the file already exists."
                ),
            },
            "content": {
                "type": "string",
                "description": "Full file content.",
            },
        },
        "required": ["path", "content"],
    }


def _edit_file_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Project-relative path of the file to edit.",
            },
            "old_text": {
                "type": "string",
                "description": "Exact text to replace. Must match the file contents.",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text.",
            },
            "count": {
                "type": "integer",
                "description": "How many occurrences to replace. Use -1 for all.",
            },
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                    "required": ["old_text", "new_text"],
                },
            },
        },
        "required": ["path"],
    }


def _read_file_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Project-relative path of the file to read.",
            },
        },
        "required": ["path"],
    }


def _read_spec_section_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "section_id": {
                "type": "string",
                "description": (
                    "The `id` of a section from the design spec (e.g. 'sec-header'). "
                    "Returns the section plus its referenced components."
                ),
            }
        },
        "required": ["section_id"],
    }


def canonical_novision_tools() -> List[CanonicalToolDefinition]:
    """The 7-tool no-vision toolset (spec §4.3).

    create_file, edit_file, read_file, list_files, finish,
    read_spec_section, read_spec_tokens.
    """
    return [
        CanonicalToolDefinition(
            name="create_file",
            description=(
                "Create (or overwrite) a file in the project tree. Use for every "
                "file you write — components, sections, config, styles. Returns "
                "a success message and the path."
            ),
            parameters=_create_file_schema(),
        ),
        CanonicalToolDefinition(
            name="edit_file",
            description=(
                "Edit an existing file using exact string replacements. Do not "
                "regenerate the entire file. Returns a success message plus a "
                "unified diff."
            ),
            parameters=_edit_file_schema(),
        ),
        CanonicalToolDefinition(
            name="read_file",
            description=(
                "Read a file you have already written. Use to self-verify "
                "consistency (imports resolve, tokens referenced, structure "
                "correct) since you cannot see a rendered preview."
            ),
            parameters=_read_file_schema(),
        ),
        CanonicalToolDefinition(
            name="list_files",
            description=(
                "List every file in the project tree so far. Use to confirm "
                "section/component coverage before finishing."
            ),
            parameters={"type": "object", "properties": {}},
        ),
        CanonicalToolDefinition(
            name="read_spec_section",
            description=(
                "Fetch a deep view of one section from the design spec — the "
                "section object plus all components it references. Call this "
                "before implementing each section so you reproduce its layout, "
                "role, and components faithfully."
            ),
            parameters=_read_spec_section_schema(),
        ),
        CanonicalToolDefinition(
            name="read_spec_tokens",
            description=(
                "Fetch the design tokens block (colors, typography scale, "
                "spacing scale). Call once at the start to generate accurate "
                "CSS variables / Tailwind theme config."
            ),
            parameters={"type": "object", "properties": {}},
        ),
        CanonicalToolDefinition(
            name="finish",
            description=(
                "Signal that the project is complete. Call once every section "
                "has a corresponding file and all tokens are applied."
            ),
            parameters={"type": "object", "properties": {}},
        ),
    ]
