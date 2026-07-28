"""No-vision tool runtime (spec §4.3).

Executes the 7-tool no-vision toolset against an in-memory project tree
(``files: Dict[str, str]``) and the design spec. Multi-file by design — unlike
``AgentToolRuntime`` which holds a single ``file_state.content``.

Vision-free import graph: imports only stdlib + ``agent.tools.types`` /
``agent.tools.summaries`` / ``agent.state``. Does NOT import
``agent.tools.runtime`` (which pulls in image-gen / screenshot_preview /
extract_assets).
"""

from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional, Tuple

from agent.state import ensure_str
from agent.tools.summaries import summarize_text
from agent.tools.types import ToolCall, ToolExecutionResult


class NovisionToolRuntime:
    """Executes no-vision tools against a multi-file project + spec."""

    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec
        self.files: Dict[str, str] = {}
        # Pre-index sections and components for fast lookup.
        self._sections_by_id: Dict[str, Dict[str, Any]] = {
            ensure_str(s.get("id")): s for s in spec.get("sections", []) if s.get("id")
        }
        self._components_by_name: Dict[str, Dict[str, Any]] = {
            ensure_str(c.get("name")): c
            for c in spec.get("components", [])
            if c.get("name")
        }

    async def execute(self, tool_call: ToolCall) -> ToolExecutionResult:
        if "INVALID_JSON" in tool_call.arguments:
            return ToolExecutionResult(
                ok=False,
                result={
                    "error": "Tool arguments were invalid JSON.",
                    "INVALID_JSON": ensure_str(tool_call.arguments.get("INVALID_JSON")),
                },
                summary={"error": "Invalid JSON tool arguments"},
            )

        handler = {
            "create_file": self._create_file,
            "edit_file": self._edit_file,
            "read_file": self._read_file,
            "list_files": self._list_files,
            "read_spec_section": self._read_spec_section,
            "read_spec_tokens": self._read_spec_tokens,
            "finish": self._finish,
        }.get(tool_call.name)

        if handler is None:
            return ToolExecutionResult(
                ok=False,
                result={"error": f"Unknown tool: {tool_call.name}"},
                summary={"error": f"Unknown tool: {tool_call.name}"},
            )
        return handler(tool_call.arguments)

    # ------------------------------------------------------------------
    # File tools
    # ------------------------------------------------------------------

    def _create_file(self, args: Dict[str, Any]) -> ToolExecutionResult:
        path = ensure_str(args.get("path"))
        content = ensure_str(args.get("content"))
        if not path:
            return ToolExecutionResult(
                ok=False,
                result={"error": "create_file requires a path"},
                summary={"error": "Missing path"},
            )
        if not content:
            return ToolExecutionResult(
                ok=False,
                result={"error": "create_file requires non-empty content"},
                summary={"error": "Missing content"},
            )
        self.files[path] = content
        summary = {
            "path": path,
            "contentLength": len(content),
            "preview": summarize_text(content, 320),
        }
        return ToolExecutionResult(
            ok=True,
            result={
                "content": f"Successfully created file at {path}.",
                "details": {"path": path, "contentLength": len(content)},
            },
            summary=summary,
        )

    def _edit_file(self, args: Dict[str, Any]) -> ToolExecutionResult:
        path = ensure_str(args.get("path"))
        if not path:
            return ToolExecutionResult(
                ok=False,
                result={"error": "edit_file requires a path"},
                summary={"error": "Missing path"},
            )
        if path not in self.files:
            return ToolExecutionResult(
                ok=False,
                result={"error": f"File not found: {path}. Call create_file first."},
                summary={"error": f"File not found: {path}"},
            )

        edits = args.get("edits")
        if not edits:
            old_text = ensure_str(args.get("old_text"))
            new_text = ensure_str(args.get("new_text"))
            count = args.get("count")
            edits = [{"old_text": old_text, "new_text": new_text, "count": count}]

        if not isinstance(edits, list):
            return ToolExecutionResult(
                ok=False,
                result={"error": "edits must be a list"},
                summary={"error": "Invalid edits payload"},
            )

        content = self.files[path]
        original_content = content
        summary_edits: List[Dict[str, Any]] = []
        for edit in edits:
            old_text = ensure_str(edit.get("old_text"))
            new_text = ensure_str(edit.get("new_text"))
            count = edit.get("count")
            if not old_text:
                return ToolExecutionResult(
                    ok=False,
                    result={"error": "edit_file requires old_text"},
                    summary={"error": "Missing old_text"},
                )
            content, replaced = self._apply_single_edit(content, old_text, new_text, count)
            if replaced == 0:
                return ToolExecutionResult(
                    ok=False,
                    result={"error": "old_text not found", "old_text": old_text},
                    summary={
                        "error": "old_text not found",
                        "old_text": summarize_text(old_text, 160),
                    },
                )
            summary_edits.append(
                {
                    "old_text": summarize_text(old_text, 140),
                    "new_text": summarize_text(new_text, 140),
                    "replaced": replaced,
                }
            )

        self.files[path] = content
        diff_info = self._generate_diff(original_content, content, path)
        return ToolExecutionResult(
            ok=True,
            result={
                "content": f"Successfully edited file at {path}.",
                "details": {
                    "diff": diff_info["diff"],
                    "firstChangedLine": diff_info["firstChangedLine"],
                },
            },
            summary={
                "path": path,
                "edits": summary_edits,
                "contentLength": len(content),
                "diff": diff_info["diff"],
                "firstChangedLine": diff_info["firstChangedLine"],
            },
        )

    def _read_file(self, args: Dict[str, Any]) -> ToolExecutionResult:
        path = ensure_str(args.get("path"))
        if not path:
            return ToolExecutionResult(
                ok=False,
                result={"error": "read_file requires a path"},
                summary={"error": "Missing path"},
            )
        if path not in self.files:
            return ToolExecutionResult(
                ok=False,
                result={"error": f"File not found: {path}"},
                summary={"error": f"File not found: {path}"},
            )
        content = self.files[path]
        return ToolExecutionResult(
            ok=True,
            result={"path": path, "content": content},
            summary={
                "path": path,
                "contentLength": len(content),
                "preview": summarize_text(content, 320),
            },
        )

    def _list_files(self, _args: Dict[str, Any]) -> ToolExecutionResult:
        paths = sorted(self.files.keys())
        return ToolExecutionResult(
            ok=True,
            result={"files": paths, "count": len(paths)},
            summary={"fileCount": len(paths), "files": paths},
        )

    def _finish(self, _args: Dict[str, Any]) -> ToolExecutionResult:
        return ToolExecutionResult(
            ok=True,
            result={"content": "Project marked as complete."},
            summary={"fileCount": len(self.files)},
        )

    # ------------------------------------------------------------------
    # Spec tools
    # ------------------------------------------------------------------

    def _read_spec_section(self, args: Dict[str, Any]) -> ToolExecutionResult:
        section_id = ensure_str(args.get("section_id"))
        if not section_id:
            return ToolExecutionResult(
                ok=False,
                result={"error": "read_spec_section requires section_id"},
                summary={"error": "Missing section_id"},
            )
        section = self._sections_by_id.get(section_id)
        if section is None:
            available = list(self._sections_by_id.keys())
            return ToolExecutionResult(
                ok=False,
                result={
                    "error": f"Section not found: {section_id}",
                    "available_section_ids": available,
                },
                summary={
                    "error": f"Section not found: {section_id}",
                    "available": available,
                },
            )
        # Resolve referenced components.
        comp_names = section.get("components", []) or []
        components = [
            self._components_by_name[name]
            for name in comp_names
            if name in self._components_by_name
        ]
        payload = {"section": section, "components": components}
        return ToolExecutionResult(
            ok=True,
            result=payload,
            summary={
                "section_id": section_id,
                "role": section.get("role"),
                "componentCount": len(components),
            },
        )

    def _read_spec_tokens(self, _args: Dict[str, Any]) -> ToolExecutionResult:
        tokens = self.spec.get("tokens", {})
        if not tokens:
            return ToolExecutionResult(
                ok=False,
                result={"error": "Spec has no tokens block"},
                summary={"error": "No tokens in spec"},
            )
        return ToolExecutionResult(
            ok=True,
            result={"tokens": tokens},
            summary={"tokenKeys": list(tokens.keys())},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_single_edit(
        content: str,
        old_text: str,
        new_text: str,
        count: Optional[int],
    ) -> Tuple[str, int]:
        if old_text not in content:
            return content, 0
        if count is None:
            replace_count = 1
        elif count < 0:
            replace_count = content.count(old_text)
        else:
            replace_count = count
        updated = content.replace(old_text, new_text, replace_count)
        return updated, min(replace_count, content.count(old_text))

    @staticmethod
    def _generate_diff(old_content: str, new_content: str, path: str) -> Dict[str, Any]:
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(old_lines, new_lines, fromfile=path, tofile=path)
        )
        diff_str = "".join(diff_lines)
        first_changed_line: Optional[int] = None
        for line in diff_lines:
            if not line.startswith("@@"):
                continue
            try:
                plus_part = line.split("+")[1].split("@@")[0].strip()
                first_changed_line = int(plus_part.split(",")[0])
            except (IndexError, ValueError):
                pass
            break
        return {"diff": diff_str, "firstChangedLine": first_changed_line}
