"""Quality gates for no-vision output (spec §7.1–§7.2).

Because the agent lost its visual feedback loop (screenshot_preview is gone),
these gates compensate by checking structural completeness against the spec:

  §7.1  Spec-coverage  — every section maps to ≥1 output file; every
                         reusable component has its own file.
  §7.2  Token-usage     — generated CSS/Tailwind references ≥80% of the
                         palette and all font families.
  §7.3  Import-graph    — every @/ or relative import resolves to a file
                         that exists in the project (catches missing
                         components before the build step).

`compute_coverage` in project_assembler.py already calculates the raw
percentages. This module wraps them with pass/fail thresholds and produces
a structured violation list that the manifest records and the route can
act on (flag or regenerate).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from codegen.project_assembler import compute_coverage

logger = logging.getLogger(__name__)

# Regex to detect inline React components (functional components named with
# uppercase first letter — the convention for React component functions).
# Catches both `function Navbar()` and `const Navbar = () =>` / `const Navbar = function()`.
_COMPONENT_FN_RE = re.compile(
    r"(?:export\s+)?(?:default\s+)?"
    r"(?:"
    r"function\s+"
    r"(Navbar|Header|Footer|Sidebar|SideBar|Layout|Card|Button|Modal|Dialog|"
    r"Dropdown|Menu|Tabs|Tab|Accordion|Form|Input|Select|Textarea|Checkbox|Radio|Switch|"
    r"Slider|Progress|Spinner|Loader|Skeleton|Avatar|Badge|Chip|Tag|Tooltip|Popover|"
    r"Drawer|Banner|Alert|Toast|Notification|Timeline|Table|Grid|List|Item|"
    r"Section|Container|Wrapper|Hero|Feature|Pricing|Testimonial|FAQ|Footer|"
    r"Logo|NavLink|NavItem|NavBrand|SocialLink|ThemeToggle|SearchBar|"
    r"[A-Z][a-zA-Z0-9]+)"
    r"\s*\(|"
    r"(?:const|let|var)\s+"
    r"(Navbar|Header|Footer|Sidebar|SideBar|Layout|Card|Button|Modal|Dialog|"
    r"Dropdown|Menu|Tabs|Tab|Accordion|Form|Input|Select|Textarea|Checkbox|Radio|Switch|"
    r"Slider|Progress|Spinner|Loader|Skeleton|Avatar|Badge|Chip|Tag|Tooltip|Popover|"
    r"Drawer|Banner|Alert|Toast|Notification|Timeline|Table|Grid|List|Item|"
    r"Section|Container|Wrapper|Hero|Feature|Pricing|Testimonial|FAQ|Footer|"
    r"Logo|NavLink|NavItem|NavBrand|SocialLink|ThemeToggle|SearchBar|"
    r"[A-Z][a-zA-Z0-9]+)"
    r"\s*[=:]\s*(?:\([^)]*\)\s*=>|function\s*)"
    r")",
    re.MULTILINE,
)

# Track components we've already extracted so we don't duplicate.
_extracted_components: set = set()

# Spec §7.2 threshold.
TOKEN_USAGE_FLOOR_PCT = 80.0

# Matches: import X from '@/path' / import X from './path' / import X from '../path'
_IMPORT_RE = re.compile(
    r"""(?:import\s+.*?\s+from\s+|import\s+|export\s+.*?\s+from\s+)"""
    r"""['"]([@./][^'"]+)['"]""",
    re.MULTILINE,
)

# npm packages that look like paths but aren't (e.g. "react", "next/image")
_NPM_PACKAGE_RE = re.compile(r"^[a-z@][a-z0-9-]*/", re.IGNORECASE)


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

    # ── Auto-repair: extract inline components into dedicated files ──
    # The model often inlines reusable components inside app/page.tsx
    # rather than creating separate files under components/. This pass
    # detects them, extracts them, and updates imports, so the coverage
    # gate that follows sees complete files.
    if fw != "html":
        patched = _auto_extract_missing_components(files, spec, _extracted_components)
        if patched:
            files = patched
            coverage = compute_coverage(files, spec)
            section_pct = float(coverage.get("section_coverage_pct", 0.0))
            token_pct = float(coverage.get("token_usage_pct", 0.0))

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

    # --- §7.3: import graph validation ---
    # Every @/ or relative import must resolve to a file that exists.
    # This catches missing components before the build step.
    if fw not in ("html", ""):
        import_violations = _check_import_graph(files)
        violations.extend(import_violations)

    return QualityReport(
        passed=len(violations) == 0,
        section_coverage_pct=section_pct,
        token_usage_pct=token_pct,
        violations=violations,
    )


def _check_import_graph(files: Dict[str, str]) -> List[str]:
    """Verify that all @/ and relative imports resolve to actual files.

    Returns a list of violation strings for any broken imports.
    """
    # Build a set of all file paths (normalized without leading ./)
    file_paths = set()
    for fpath in files:
        clean = fpath.lstrip("./")
        file_paths.add(clean)
        file_paths.add(fpath)

    violations: List[str] = []
    # Only check JS/TS files
    code_extensions = (".tsx", ".ts", ".jsx", ".js")
    for fpath, content in files.items():
        if not fpath.endswith(code_extensions):
            continue
        for match in _IMPORT_RE.finditer(content):
            ref = match.group(1)
            # Skip external URLs and protocols
            if ref.startswith(
                ("#", "http://", "https://", "data:", "mailto:", "tel:")
            ):
                continue
            # Skip npm packages (e.g. "react", "next/image", "lucide-react")
            if not ref.startswith(("@/", "./", "/", "../")):
                continue
            # Normalize @/ alias to root-relative
            if ref.startswith("@/"):
                clean = ref[2:]
            else:
                clean = ref.lstrip("./")
            if not clean:
                continue
            # Try exact match and common extensions
            candidates = [
                clean,
                clean + ".tsx",
                clean + ".ts",
                clean + ".jsx",
                clean + ".js",
                clean + ".css",
                clean + "/index.tsx",
                clean + "/index.ts",
            ]
            if not any(c in file_paths for c in candidates):
                violations.append(
                    f"import_graph: {fpath} imports '{ref}' "
                    f"but no matching file exists"
                )

    return violations


def _auto_extract_missing_components(
    files: Dict[str, str],
    spec: Dict[str, Any],
    extracted_set: set,
) -> Dict[str, str]:
    """Detect reusable components inlined in page.tsx and extract them.

    The model sometimes defines components inside ``app/page.tsx`` instead
    of creating dedicated files under ``components/``.  This function:

    1. Scans ``app/page.tsx`` for inline component definitions that match
       reusable component names from the spec.
    2. Extracts each into ``components/<Name>.tsx``.
    3. Replaces the definition in page.tsx with an import.
    4. Updates the extracted_set so subsequent calls skip already-extracted
       components.

    Returns a **new** dict (shallow copy of ``files``) with the fixes
    applied, or the original ``files`` unchanged if nothing to extract.
    """
    # ── 1. Gather reusable component names from spec ──
    spec_comps: Dict[str, str] = {}
    comps_raw = spec.get("components") or []
    if not isinstance(comps_raw, list):
        comps_raw = []
    for c in comps_raw:
        if isinstance(c, dict) and c.get("reusable") and c.get("name"):
            spec_comps[c["name"]] = c.get("description") or c.get("name", "")

    if not spec_comps:
        return files  # nothing to check

    # ── 2. Read page.tsx content ──
    page_key = next(
        (k for k in files if k.endswith("/page.tsx") or k == "page.tsx"),
        None,
    )
    if not page_key:
        return files
    content = files[page_key]

    # ── 3. For each spec reusable component, check if it's inlined ──
    result = content
    missing_comp_found = False

    for comp_name, _comp_desc in spec_comps.items():
        if comp_name in extracted_set:
            continue

        # Check if this component already has a dedicated file
        existing_file = any(
            k.endswith(f"/{comp_name}.tsx") or k == f"{comp_name}.tsx"
            for k in files
        )
        if existing_file:
            extracted_set.add(comp_name)
            continue

        # Try to find in page.tsx — support both
        #   function CompName(...) and const CompName = (...) =>
        fn_pat = re.compile(
            r"(?:export\s+)?(?:default\s+)?"
            rf"function\s+{comp_name}\s*\(",
            re.MULTILINE,
        )
        m = fn_pat.search(result)

        if not m:
            const_pat = re.compile(
                r"(?:export\s+)?(?:default\s+)?"
                rf"(?:const|let|var)\s+{comp_name}\s*[=:]\s*"
                r"(?:\([^)]*\)\s*=>|function\s*\()",
                re.MULTILINE,
            )
            m = const_pat.search(result)

        if not m:
            continue  # not inlined in page.tsx

        # ── Extract body by counting braces ──
        start_idx = m.start()
        body_start = result.index("{", m.end()) + 1
        depth = 1
        i = body_start
        while depth > 0 and i < len(result):
            if result[i] == "{":
                depth += 1
            elif result[i] == "}":
                depth -= 1
            i += 1
        body_end = i - 1

        component_code = result[start_idx : body_end + 1]
        if not component_code.strip().startswith("export"):
            component_code = component_code.replace(
                f"function {comp_name}",
                f"export function {comp_name}",
            )
            component_code = component_code.replace(
                f"const {comp_name}",
                f"export const {comp_name}",
            )

        import_block = _extract_import_block(result, start_idx)
        full_content = import_block + "\n" + component_code + "\n"

        # Replace definition with import in page.tsx
        replacement = f"import {{ {comp_name} }} from '@/components/{comp_name}'"
        result = result[:start_idx] + replacement + "\n" + result[body_end + 1 :]

        comp_path = f"components/{comp_name}.tsx"
        files = {**files, comp_path: full_content}
        extracted_set.add(comp_name)
        missing_comp_found = True

    if not missing_comp_found:
        return files

    files = {**files, page_key: result}
    return files


def _extract_import_block(content: str, before_offset: int) -> str:
    """Extract unique import statements from content up to *before_offset*."""
    lines = content[:before_offset].split("\n")
    imports: List[str] = []
    seen: set = set()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") and ('"' in stripped or "'" in stripped):
            if stripped not in seen:
                imports.append(line.rstrip())
                seen.add(stripped)
    return "\n".join(imports)
