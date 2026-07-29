"""Build smoke test for no-vision output (spec §7.3).

For Next/Nuxt/Astro: run `install + build` in a temp copy of the generated
project. Must compile without errors.

For HTML: open the entry file in headless Chromium via Playwright and assert
no console errors fire on load.

This is a CI-grade gate — it runs the real toolchain, so it is slow (tens of
seconds). The route calls it after assembly; a failure flags the output but
does not discard it (the manifest records the smoke result).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Frameworks that need a real install+build step.
_BUILD_FRAMEWORKS = {"next", "nuxt", "astro"}


@dataclass
class SmokeResult:
    """Result of a build smoke test."""

    framework: str
    passed: bool
    duration_s: float = 0.0
    error: Optional[str] = None
    log_tail: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "passed": self.passed,
            "duration_s": round(self.duration_s, 1),
            "error": self.error,
            "log_tail": self.log_tail,
        }


async def run_build_smoke(
    files: Dict[str, str], framework: str, timeout: int = 180
) -> SmokeResult:
    """Run the spec §7.3 build smoke test.

    Args:
        files: assembled project files (path → content).
        framework: one of "next", "nuxt", "astro", "html".
        timeout: max seconds for the install+build subprocess.

    Returns:
        A SmokeResult. Never raises — a build failure is a failed result,
        not an exception, so the route can record it in the manifest.
    """
    framework = (framework or "").lower().strip()

    if framework in _BUILD_FRAMEWORKS:
        return _smoke_build_framework(files, framework, timeout)
    elif framework == "html":
        return await _smoke_html(files)
    else:
        # Unknown framework — can't smoke-test, treat as skipped (pass).
        return SmokeResult(framework=framework or "unknown", passed=True)


# ---------------------------------------------------------------------------
# Framework install + build
# ---------------------------------------------------------------------------

def _smoke_build_framework(
    files: Dict[str, str], framework: str, timeout: int
) -> SmokeResult:
    """Materialize files into a temp dir and run install + build."""
    import time

    tmpdir = tempfile.mkdtemp(prefix=f"smoke-{framework}-")
    try:
        root = Path(tmpdir)
        for rel_path, content in files.items():
            # Guard against path traversal in generated paths.
            safe = root / rel_path
            try:
                safe.relative_to(root)
            except ValueError:
                continue
            safe.parent.mkdir(parents=True, exist_ok=True)
            safe.write_text(content, encoding="utf-8")

        # package.json must exist for install+build to mean anything.
        if not (root / "package.json").exists():
            return SmokeResult(
                framework=framework,
                passed=False,
                error="missing package.json — cannot install/build",
            )

        t0 = time.monotonic()

        # 1. install
        install_ok, install_log = _run_subprocess(
            ["npm", "install", "--no-audit", "--no-fund"], root, timeout
        )
        if not install_ok:
            return SmokeResult(
                framework=framework,
                passed=False,
                duration_s=time.monotonic() - t0,
                error="npm install failed",
                log_tail=install_log[-15:],
            )

        # 2. build
        build_ok, build_log = _run_subprocess(
            ["npm", "run", "build"], root, timeout
        )
        elapsed = time.monotonic() - t0
        return SmokeResult(
            framework=framework,
            passed=build_ok,
            duration_s=elapsed,
            error=None if build_ok else "npm run build failed",
            log_tail=build_log[-15:],
        )
    except Exception as exc:  # never raise out of the smoke gate
        logger.exception("build smoke crashed for %s", framework)
        return SmokeResult(framework=framework, passed=False, error=str(exc))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _run_subprocess(
    cmd: List[str], cwd: Path, timeout: int
) -> tuple[bool, List[str]]:
    """Run a subprocess, capture output lines. Returns (success, log_lines)."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        log = (proc.stdout + "\n" + proc.stderr).splitlines()
        return proc.returncode == 0, log
    except subprocess.TimeoutExpired:
        return False, [f"TIMEOUT after {timeout}s"]
    except FileNotFoundError:
        return False, [f"command not found: {cmd[0]}"]


# ---------------------------------------------------------------------------
# HTML headless console-error check
# ---------------------------------------------------------------------------

async def _smoke_html(files: Dict[str, str]) -> SmokeResult:
    """Open the HTML entry in headless Chromium, assert no console errors."""
    import time

    # Find the entry HTML file.
    entry = None
    for path in files:
        if path.lower().endswith(".html") and "index" in path.lower():
            entry = path
            break
    if entry is None:
        # Fall back to any .html
        htmls = [p for p in files if p.lower().endswith(".html")]
        if not htmls:
            return SmokeResult(
                framework="html", passed=False, error="no .html entry file found"
            )
        entry = htmls[0]

    tmpdir = tempfile.mkdtemp(prefix="smoke-html-")
    try:
        root = Path(tmpdir)
        for rel_path, content in files.items():
            safe = root / rel_path
            try:
                safe.relative_to(root)
            except ValueError:
                continue
            safe.parent.mkdir(parents=True, exist_ok=True)
            safe.write_text(content, encoding="utf-8")

        entry_abs = (root / entry).resolve()
        t0 = time.monotonic()

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return SmokeResult(
                framework="html",
                passed=False,
                error="playwright not installed",
            )

        console_errors: List[str] = []
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                page = await browser.new_page()
                page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
                page.on("pageerror", lambda err: console_errors.append(str(err)))
                await page.goto(f"file://{entry_abs}", wait_until="networkidle", timeout=15000)
                await page.wait_for_timeout(1000)  # let deferred scripts settle
                await browser.close()
        except Exception as exc:
            return SmokeResult(
                framework="html",
                passed=False,
                duration_s=time.monotonic() - t0,
                error=f"playwright failed: {exc}",
            )

        elapsed = time.monotonic() - t0
        if console_errors:
            return SmokeResult(
                framework="html",
                passed=False,
                duration_s=elapsed,
                error=f"{len(console_errors)} console error(s)",
                log_tail=console_errors[:10],
            )
        return SmokeResult(framework="html", passed=True, duration_s=elapsed)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
