"""Project assembler for the no-vision path (spec §5).

Takes the agent's raw file map (``{path: content}``) + the input spec and:
  1. Computes coverage metrics (spec §7.1, §7.2).
  2. Builds a sidecar ``manifest.json``.
  3. Packages everything into a zip.

The agent writes files with project-relative paths already (e.g.
``src/components/Button.tsx``), so assembly is mostly packaging + metrics.
We do NOT rewrite the agent's file tree — that would second-guess the model.
We only validate and annotate.

Vision-free import graph: stdlib only.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from typing import Any, Dict, List, Optional


def assemble_project(
    files: Dict[str, str],
    spec: Dict[str, Any],
    framework: str,
    stack: str,
    generation_meta: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Assemble a multi-file project zip with a manifest (spec §5.2).

    Returns the raw zip bytes.
    """
    coverage = compute_coverage(files, spec)
    manifest = build_manifest(
        spec=spec,
        framework=framework,
        stack=stack,
        coverage=coverage,
        generation_meta=generation_meta or {},
        file_count=len(files),
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, content in sorted(files.items()):
            zf.writestr(path, content)
        zf.writestr("manifest.json", json.dumps(manifest_safe(manifest), indent=2))
    return buf.getvalue()


def build_manifest(
    spec: Dict[str, Any],
    framework: str,
    stack: str,
    coverage: Dict[str, Any],
    generation_meta: Dict[str, Any],
    file_count: int,
) -> Dict[str, Any]:
    return {
        "spec_version": str(spec.get("version", "unknown")),
        "source_url": spec.get("url"),
        "source_title": spec.get("title"),
        "framework": framework,
        "stack": stack,
        "file_count": file_count,
        "coverage": coverage,
        "generation": generation_meta,
    }


def manifest_safe(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure manifest is JSON-serializable (strip non-serializable values)."""
    return json.loads(json.dumps(manifest, default=str))


# ---------------------------------------------------------------------------
# Coverage metrics (spec §7.1, §7.2)
# ---------------------------------------------------------------------------

def compute_coverage(files: Dict[str, str], spec: Dict[str, Any]) -> Dict[str, Any]:
    """Compute spec-coverage and token-usage metrics.

    - section_coverage (spec §7.1): fraction of sections that map to ≥1 file.
    - component_coverage (spec §7.1): fraction of reusable components with own file.
    - token_usage_pct (spec §7.2): % of palette colors + font families referenced.
    """
    all_content = "\n".join(files.values())
    file_paths = set(files.keys())

    # --- section coverage ---
    sections = spec.get("sections", []) or []
    section_hits = 0
    section_details: List[Dict[str, Any]] = []
    for section in sections:
        sid = str(section.get("id", ""))
        role = str(section.get("role", ""))
        # A section is "covered" if any file path or content references its id or role.
        covered = _section_is_covered(sid, role, file_paths, all_content)
        if covered:
            section_hits += 1
        section_details.append({"id": sid, "role": role, "covered": covered})
    section_pct = round(100 * section_hits / len(sections), 1) if sections else 0.0

    # --- reusable component coverage ---
    components = spec.get("components", []) or []
    reusable = [c for c in components if c.get("reusable")]
    reusable_hits = 0
    component_details: List[Dict[str, Any]] = []
    for comp in reusable:
        name = str(comp.get("name", ""))
        covered = _component_has_file(name, file_paths)
        if covered:
            reusable_hits += 1
        component_details.append({"name": name, "covered": covered})
    component_pct = (
        round(100 * reusable_hits / len(reusable), 1) if reusable else 100.0
    )

    # --- token usage (spec §7.2: ≥80% of palette + all font families) ---
    tokens = spec.get("tokens", {}) or {}
    palette = _extract_palette_hex(tokens)
    fonts = _extract_font_families(tokens)
    palette_hits = sum(1 for hex_val in palette if _hex_referenced(hex_val, all_content))
    font_hits = sum(1 for font in fonts if _font_referenced(font, all_content))
    palette_total = len(palette) or 1
    font_total = len(fonts) or 1
    # Combined percentage: average of palette and font coverage.
    token_usage_pct = round(
        100 * (palette_hits + font_hits) / (palette_total + font_total), 1
    )

    return {
        "section_coverage_pct": section_pct,
        "section_covered": section_hits,
        "section_total": len(sections),
        "section_details": section_details,
        "reusable_component_coverage_pct": component_pct,
        "reusable_component_covered": reusable_hits,
        "reusable_component_total": len(reusable),
        "component_details": component_details,
        "token_usage_pct": token_usage_pct,
        "palette_referenced": palette_hits,
        "palette_total": palette_total,
        "fonts_referenced": font_hits,
        "fonts_total": font_total,
    }


# ---------------------------------------------------------------------------
# Coverage helpers
# ---------------------------------------------------------------------------

def _section_is_covered(
    section_id: str,
    role: str,
    file_paths: set,
    all_content: str,
) -> bool:
    """A section is covered if a file path or content mentions its id or role."""
    if not section_id and not role:
        return False
    # Check file paths first (e.g. src/sections/Header.tsx).
    for path in file_paths:
        lower = path.lower()
        if section_id and section_id.lower() in lower:
            return True
        if role and role.lower() in lower:
            return True
    # Then check content (e.g. a comment or data-section attribute).
    lower_content = all_content.lower()
    if section_id and section_id.lower() in lower_content:
        return True
    if role and role.lower() in lower_content:
        return True
    return False


def _component_has_file(name: str, file_paths: set) -> bool:
    """A reusable component is covered if a file is named after it (PascalCase or kebab)."""
    if not name:
        return False
    pascal = _to_pascal(name)
    kebab = _to_kebab(name)
    for path in file_paths:
        lower = path.lower()
        basename = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        if name.lower() in basename or pascal.lower() in basename or kebab in lower:
            return True
    return False


def _extract_palette_hex(tokens: Dict[str, Any]) -> List[str]:
    colors = tokens.get("colors", {}) or {}
    palette = colors.get("palette", []) or []
    hexes: List[str] = []
    for entry in palette:
        h = entry.get("hex") if isinstance(entry, dict) else None
        if h:
            hexes.append(str(h))
    return hexes


def _extract_font_families(tokens: Dict[str, Any]) -> List[str]:
    typo = tokens.get("typography", {}) or {}
    return [str(f) for f in typo.get("font_families", []) or []]


def _hex_referenced(hex_val: str, content: str) -> bool:
    """True if the hex value (case-insensitive) appears in the generated content."""
    return hex_val.lower() in content.lower()


def _font_referenced(font: str, content: str) -> bool:
    """True if the font family name appears in the generated content."""
    return font.lower() in content.lower()


def _to_pascal(name: str) -> str:
    parts = re.split(r"[-_\s]+", name)
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def _to_kebab(name: str) -> str:
    return re.sub(r"[-_\s]+", "-", name).strip("-").lower()
