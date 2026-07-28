"""Tests for the project assembler (spec §5, §7.1, §7.2).

Covers: zip packaging, manifest correctness, framework parity (HTML/Next/Nuxt/
Astro), section coverage, reusable-component coverage, and token-usage metrics.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any, Dict

import pytest

from codegen.project_assembler import assemble_project, compute_coverage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SPEC: Dict[str, Any] = {
    "version": "1.0",
    "url": "https://example.com",
    "title": "Test",
    "tokens": {
        "colors": {
            "palette": [
                {"hex": "#0b0f19", "role": "bg"},
                {"hex": "#6366f1", "role": "accent"},
                {"hex": "#e5e7eb", "role": "text"},
            ]
        },
        "typography": {"font_families": ["Inter", "JetBrains Mono"]},
        "spacing": {"scale": [{"name": "md", "value": "16px"}]},
    },
    "sections": [
        {"id": "sec-header", "role": "header", "components": ["logo"]},
        {"id": "sec-hero", "role": "hero", "components": ["cta-button"]},
        {"id": "sec-footer", "role": "footer", "components": []},
    ],
    "components": [
        {"name": "cta-button", "type": "button", "reusable": True},
        {"name": "logo", "type": "image", "reusable": True},
        {"name": "hero-text", "type": "text", "reusable": False},
    ],
}


def _files_covering_most() -> Dict[str, str]:
    """A file map that covers 2/3 sections, 2/2 reusable components, most tokens."""
    return {
        "src/sections/Header.tsx": "// section sec-header\nexport default () => null",
        "src/sections/Hero.tsx": "// section sec-hero\nexport default () => null",
        "src/components/CtaButton.tsx": "/* cta-button */ color: #6366f1;",
        "src/components/Logo.tsx": "/* logo */",
        "src/styles/tokens.css": ":root { --bg: #0b0f19; --text: #e5e7eb; font-family: Inter; }",
        "tailwind.config.js": "module.exports = { theme: { extend: { colors: { accent: '#6366f1' } } } }",
    }


# ---------------------------------------------------------------------------
# Packaging + manifest
# ---------------------------------------------------------------------------

def test_assemble_produces_valid_zip_with_manifest() -> None:
    files = _files_covering_most()
    zip_bytes = assemble_project(files, SPEC, framework="next", stack="tailwind")
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = zf.namelist()
    # All input files present.
    for path in files:
        assert path in names, f"Missing {path} in zip"
    # Manifest present and valid.
    assert "manifest.json" in names
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["framework"] == "next"
    assert manifest["stack"] == "tailwind"
    assert manifest["spec_version"] == "1.0"
    assert manifest["source_url"] == "https://example.com"
    assert manifest["file_count"] == len(files)
    assert "coverage" in manifest
    assert "generation" in manifest


def test_manifest_includes_generation_meta() -> None:
    files = {"index.html": "<html></html>"}
    zip_bytes = assemble_project(
        files,
        SPEC,
        framework="html",
        stack="html_css",
        generation_meta={"finished": True, "iterations": 5, "tool_call_count": 12},
    )
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["generation"]["iterations"] == 5
    assert manifest["generation"]["tool_call_count"] == 12


# ---------------------------------------------------------------------------
# Framework parity (spec §5.3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("framework", ["html", "next", "nuxt", "astro"])
def test_assemble_works_for_all_frameworks(framework: str) -> None:
    """The assembler must handle all 4 frameworks (spec §5.3)."""
    files = {f"src/index.{_ext(framework)}": "content"}
    zip_bytes = assemble_project(files, SPEC, framework=framework, stack="tailwind")
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["framework"] == framework


def _ext(framework: str) -> str:
    return {"html": "html", "next": "tsx", "nuxt": "vue", "astro": "astro"}[framework]


# ---------------------------------------------------------------------------
# Coverage metrics (spec §7.1, §7.2)
# ---------------------------------------------------------------------------

def test_section_coverage_counts_hits() -> None:
    files = _files_covering_most()
    cov = compute_coverage(files, SPEC)
    # Header + Hero are covered (files named after role); Footer is not.
    assert cov["section_total"] == 3
    assert cov["section_covered"] == 2
    assert cov["section_coverage_pct"] == pytest.approx(66.7, abs=0.1)


def test_section_coverage_via_content_reference() -> None:
    """A section referenced only in content (not filename) still counts."""
    files = {"index.html": '<footer data-section="sec-footer"></footer>'}
    cov = compute_coverage(files, SPEC)
    footer = [s for s in cov["section_details"] if s["id"] == "sec-footer"][0]
    assert footer["covered"] is True


def test_reusable_component_coverage() -> None:
    files = _files_covering_most()
    cov = compute_coverage(files, SPEC)
    # cta-button → CtaButton.tsx, logo → Logo.tsx. Both covered.
    assert cov["reusable_component_total"] == 2
    assert cov["reusable_component_covered"] == 2
    assert cov["reusable_component_coverage_pct"] == 100.0


def test_reusable_component_kebab_case_match() -> None:
    """Component 'cta-button' matches file 'cta-button.tsx' (kebab)."""
    files = {"src/components/cta-button.tsx": "x"}
    cov = compute_coverage(files, SPEC)
    cta = [c for c in cov["component_details"] if c["name"] == "cta-button"][0]
    assert cta["covered"] is True


def test_token_usage_palette_and_fonts() -> None:
    files = _files_covering_most()
    cov = compute_coverage(files, SPEC)
    # Palette: #0b0f19 (tokens.css), #6366f1 (CtaButton + tailwind.config), #e5e7eb (tokens.css) → 3/3
    assert cov["palette_total"] == 3
    assert cov["palette_referenced"] == 3
    # Fonts: Inter (tokens.css) → 1/2 (JetBrains Mono not referenced)
    assert cov["fonts_total"] == 2
    assert cov["fonts_referenced"] == 1
    # Combined: (3+1)/(3+2) = 80%
    assert cov["token_usage_pct"] == 80.0


def test_empty_files_yield_zero_coverage() -> None:
    cov = compute_coverage({}, SPEC)
    assert cov["section_coverage_pct"] == 0.0
    assert cov["section_covered"] == 0
    assert cov["token_usage_pct"] == 0.0


def test_empty_spec_does_not_crash() -> None:
    """compute_coverage must tolerate a spec with no sections/components/tokens."""
    cov = compute_coverage({"index.html": "x"}, {"version": "1.0"})
    assert cov["section_total"] == 0
    assert cov["section_coverage_pct"] == 0.0
    assert cov["reusable_component_coverage_pct"] == 100.0  # no reusable → vacuously full
    assert cov["token_usage_pct"] == 0.0
