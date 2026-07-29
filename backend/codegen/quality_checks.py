"""Quality gates for no-vision output (spec §7.1–§7.2).

Because the agent lost its visual feedback loop (screenshot_preview is gone),
these gates compensate by checking structural completeness against the spec:

  §7.1  Spec-coverage  — every section maps to ≥1 output file; every
                         reusable component has its own file.
  §7.2  Token-usage     — generated CSS/Tailwind references ≥80% of the
                         palette and all font families.

`compute_coverage` in project_assembler.py already calculates the raw
percentages. This module wraps them with pass/fail thresholds and produces
a structured violation list that the manifest records and the route can
act on (flag or regenerate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from codegen.project_assembler import compute_coverage

# Spec §7.2 threshold.
TOKEN_USAGE_FLOOR_PCT = 80.0


@dataclass
class QualityReport:
    """Result of running the quality gates on assembled output."""

    passed: bool
    section_coverage_pct: float
    token_usage_pct: float
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "section_coverage_pct": self.section_coverage_pct,
            "token_usage_pct": self.token_usage_pct,
            "violations": self.violations,
        }


def run_quality_checks(
    files: Dict[str, str], spec: Dict[str, Any], framework: str = ""
) -> QualityReport:
    """Run spec §7.1 + §7.2 gates against assembled output.

    Args:
        files: assembled project files (path → content).
        spec: the input design spec (version "1.0").
        framework: target framework (e.g. "html", "next"). For "html",
            component file coverage is skipped because components are inlined.

    Returns:
        A QualityReport with pass/fail and a list of human-readable violations.
    """
    coverage = compute_coverage(files, spec)
    section_pct = float(coverage.get("section_coverage_pct", 0.0))
    token_pct = float(coverage.get("token_usage_pct", 0.0))
    fw = (framework or "").lower().strip()

    violations: List[str] = []

    # --- §7.1: section coverage ---
    section_details = coverage.get("section_details", [])
    missing_sections = [
        d.get("id") or d.get("role", "?")
        for d in section_details
        if not d.get("covered")
    ]
    if missing_sections:
        violations.append(
            f"section_coverage: {len(missing_sections)} section(s) with no "
            f"output file: {', '.join(missing_sections[:5])}"
            + (" …" if len(missing_sections) > 5 else "")
        )

    # --- §7.1: reusable component coverage (skip for static HTML) ---
    # In HTML framework, components are inlined as CSS class patterns in
    # index.html — they don't get separate files, so this check is N/A.
    if fw != "html":
        component_details = coverage.get("component_details", [])
        missing_components = [
            d.get("name", "?")
            for d in component_details
            if not d.get("covered")
        ]
        if missing_components:
            violations.append(
                f"component_coverage: {len(missing_components)} reusable "
                f"component(s) missing own file: {', '.join(missing_components[:5])}"
                + (" …" if len(missing_components) > 5 else "")
            )

    # --- §7.2: token usage ---
    if token_pct < TOKEN_USAGE_FLOOR_PCT:
        palette_total = int(coverage.get("palette_total", 0))
        palette_referenced = int(coverage.get("palette_referenced", 0))
        fonts_total = int(coverage.get("fonts_total", 0))
        fonts_referenced = int(coverage.get("fonts_referenced", 0))
        missing_count = (palette_total - palette_referenced) + (
            fonts_total - fonts_referenced
        )
        violations.append(
            f"token_usage: {token_pct:.1f}% < {TOKEN_USAGE_FLOOR_PCT:.0f}% floor "
            f"({missing_count} unreferenced palette/font token(s))"
        )

    return QualityReport(
        passed=len(violations) == 0,
        section_coverage_pct=section_pct,
        token_usage_pct=token_pct,
        violations=violations,
    )
