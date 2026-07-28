"""Tests for quality gates (spec §7.1–§7.2).

Covers: section coverage pass/fail, reusable-component coverage pass/fail,
token-usage floor (80%), and the combined pass case. Uses the same SPEC
shape as test_project_assembler.py so coverage semantics stay consistent.
"""

from __future__ import annotations

from typing import Any, Dict

from codegen.quality_checks import (
    TOKEN_USAGE_FLOOR_PCT,
    QualityReport,
    run_quality_checks,
)


SPEC: Dict[str, Any] = {
    "version": "1.0",
    "url": "https://example.com",
    "title": "Test Page",
    "tokens": {
        "colors": {
            "palette": [
                {"hex": "#0b0f19", "role": "bg"},
                {"hex": "#e5e7eb", "role": "text"},
                {"hex": "#6366f1", "role": "accent"},
            ]
        },
        "typography": {"font_families": ["Inter"]},
    },
    "sections": [
        {"id": "sec-header", "role": "header", "components": []},
        {"id": "sec-hero", "role": "hero", "components": ["cta-button"]},
        {"id": "sec-footer", "role": "footer", "components": []},
    ],
    "components": [
        {"name": "cta-button", "type": "button", "reusable": True},
        {"name": "logo", "type": "image", "reusable": True},
        {"name": "hero-text", "type": "text", "reusable": False},
    ],
}


def _full_coverage_files() -> Dict[str, str]:
    """Files that cover all 3 sections, both reusable components, and all tokens."""
    return {
        "src/sections/Header.tsx": "// section sec-header\nexport default () => null",
        "src/sections/Hero.tsx": "// section sec-hero\nexport default () => null",
        "src/sections/Footer.tsx": "// section sec-footer\nexport default () => null",
        "src/components/CtaButton.tsx": "/* cta-button */ color: #6366f1;",
        "src/components/Logo.tsx": "/* logo */",
        "src/styles/tokens.css": (
            ":root { --bg: #0b0f19; --text: #e5e7eb; font-family: Inter; }"
        ),
        "tailwind.config.js": (
            "module.exports = { theme: { extend: { colors: { accent: '#6366f1' } } } }"
        ),
    }


def test_full_coverage_passes() -> None:
    report = run_quality_checks(_full_coverage_files(), SPEC)
    assert isinstance(report, QualityReport)
    assert report.passed is True
    assert report.violations == []
    assert report.section_coverage_pct == 100.0
    assert report.token_usage_pct >= TOKEN_USAGE_FLOOR_PCT


def test_missing_section_fails() -> None:
    files = _full_coverage_files()
    del files["src/sections/Footer.tsx"]  # drop one section's only file
    report = run_quality_checks(files, SPEC)
    assert report.passed is False
    assert any("section_coverage" in v for v in report.violations)
    assert "sec-footer" in " ".join(report.violations)


def test_missing_component_fails() -> None:
    files = _full_coverage_files()
    del files["src/components/Logo.tsx"]  # drop a reusable component's file
    report = run_quality_checks(files, SPEC)
    assert report.passed is False
    assert any("component_coverage" in v for v in report.violations)
    assert "logo" in " ".join(report.violations)


def test_low_token_usage_fails() -> None:
    """Files that reference none of the palette/fonts trip the §7.2 floor."""
    files = {
        "src/sections/Header.tsx": "// section sec-header",
        "src/sections/Hero.tsx": "// section sec-hero",
        "src/sections/Footer.tsx": "// section sec-footer",
        "src/components/CtaButton.tsx": "/* cta-button */",
        "src/components/Logo.tsx": "/* logo */",
        # No tokens.css, no tailwind config → 0% token usage.
    }
    report = run_quality_checks(files, SPEC)
    assert report.passed is False
    assert any("token_usage" in v for v in report.violations)
    assert report.token_usage_pct < TOKEN_USAGE_FLOOR_PCT


def test_to_dict_shape() -> None:
    report = run_quality_checks(_full_coverage_files(), SPEC)
    d = report.to_dict()
    assert set(d.keys()) == {
        "passed",
        "section_coverage_pct",
        "token_usage_pct",
        "violations",
    }
    assert d["passed"] is True
    assert isinstance(d["violations"], list)


def test_empty_files_fails() -> None:
    """No files at all → every gate fails."""
    report = run_quality_checks({}, SPEC)
    assert report.passed is False
    # At least section + token violations (components may or may not be flagged
    # depending on coverage semantics, so don't assert on it).
    assert any("section_coverage" in v for v in report.violations)
