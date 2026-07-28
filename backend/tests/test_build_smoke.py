"""Tests for the build smoke gate (spec §7.3).

The smoke gate shells out to npm (Next/Nuxt/Astro) or Playwright (HTML), so
these tests mock the subprocess and Playwright layers to stay hermetic — no
real toolchain or browser is required. The pure-logic branches (unknown
framework, missing package.json) are exercised directly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

from codegen import build_smoke
from codegen.build_smoke import SmokeResult, run_build_smoke


# ---------------------------------------------------------------------------
# Pure-logic branches (no subprocess, no browser)
# ---------------------------------------------------------------------------

def test_unknown_framework_passes() -> None:
    """An unrecognized framework can't be smoke-tested → treated as skipped (pass)."""
    report = run_build_smoke({"index.html": "<html></html>"}, "vue")
    assert report.passed is True
    assert report.framework == "vue"


def test_empty_framework_string_passes() -> None:
    report = run_build_smoke({}, "")
    assert report.passed is True
    assert report.framework == "unknown"


def test_smoke_result_to_dict_shape() -> None:
    r = SmokeResult(framework="next", passed=False, error="boom", log_tail=["a", "b"])
    d = r.to_dict()
    assert d == {
        "framework": "next",
        "passed": False,
        "duration_s": 0.0,
        "error": "boom",
        "log_tail": ["a", "b"],
    }


# ---------------------------------------------------------------------------
# Framework install + build (subprocess mocked)
# ---------------------------------------------------------------------------

def _fake_subprocess_runner(outcomes: List[Tuple[int, List[str]]]):
    """Build a fake _run_subprocess that returns each (rc, log) in sequence."""
    it = iter(outcomes)

    def _fake(cmd: List[str], cwd: Path, timeout: int) -> Tuple[bool, List[str]]:
        rc, log = next(it)
        return rc == 0, log

    return _fake


def test_next_build_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    files = {"package.json": "{}", "src/app/page.tsx": "export default () => null"}
    monkeypatch.setattr(
        build_smoke,
        "_run_subprocess",
        _fake_subprocess_runner([(0, ["install ok"]), (0, ["build ok"])]),
    )
    report = run_build_smoke(files, "next")
    assert report.passed is True
    assert report.framework == "next"
    assert report.error is None


def test_next_install_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    files = {"package.json": "{}"}
    monkeypatch.setattr(
        build_smoke,
        "_run_subprocess",
        _fake_subprocess_runner([(1, ["npm ERR! ERESOLVE"])]) ,
    )
    report = run_build_smoke(files, "next")
    assert report.passed is False
    assert report.error == "npm install failed"
    assert "npm ERR! ERESOLVE" in report.log_tail


def test_next_build_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    files = {"package.json": "{}"}
    monkeypatch.setattr(
        build_smoke,
        "_run_subprocess",
        _fake_subprocess_runner([(0, ["install ok"]), (1, ["Type error: X"])]),
    )
    report = run_build_smoke(files, "next")
    assert report.passed is False
    assert report.error == "npm run build failed"
    assert "Type error: X" in report.log_tail


def test_missing_package_json_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A framework project with no package.json can't build."""
    # _run_subprocess should never be called.
    monkeypatch.setattr(
        build_smoke, "_run_subprocess", lambda *a, **k: pytest.fail("should not build")
    )
    files = {"README.md": "no package.json here"}
    report = run_build_smoke(files, "astro")
    assert report.passed is False
    assert "package.json" in (report.error or "")


def test_subprocess_timeout() -> None:
    """_run_subprocess catches TimeoutExpired and returns a failed result."""
    # Patch subprocess.run to raise TimeoutExpired; _run_subprocess should
    # catch it and return (False, ["TIMEOUT after ..."]).
    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd=["npm"], timeout=180)
    with patch.object(build_smoke.subprocess, "run", side_effect=_raise):
        ok, log = build_smoke._run_subprocess(["npm", "install"], Path("/tmp"), 180)
    assert ok is False
    assert "TIMEOUT" in log[-1]


def test_subprocess_command_not_found() -> None:
    """A missing binary surfaces as a failed result, not an exception."""
    def _raise(*a, **k):
        raise FileNotFoundError(2, "No such file", "npm")
    with patch.object(build_smoke.subprocess, "run", side_effect=_raise):
        ok, log = build_smoke._run_subprocess(["npm", "install"], Path("/tmp"), 180)
    assert ok is False
    assert "command not found" in log[-1]


def test_build_smoke_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even an internal crash is caught and returned as a failed result."""
    def _boom(cmd, cwd, timeout):
        raise RuntimeError("internal boom")
    monkeypatch.setattr(build_smoke, "_run_subprocess", _boom)
    files = {"package.json": "{}"}
    report = run_build_smoke(files, "next")
    assert report.passed is False
    assert "internal boom" in (report.error or "")


# ---------------------------------------------------------------------------
# HTML headless console-error check (Playwright mocked)
# ---------------------------------------------------------------------------

def test_html_no_console_errors_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    files = {"index.html": "<html><body>hi</body></html>"}
    _patch_playwright(monkeypatch, console_errors=[], page_errors=[])
    report = run_build_smoke(files, "html")
    assert report.passed is True
    assert report.framework == "html"


def test_html_console_errors_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    files = {"index.html": "<html><body>hi</body></html>"}
    _patch_playwright(
        monkeypatch, console_errors=["Uncaught TypeError: X is undefined"], page_errors=[]
    )
    report = run_build_smoke(files, "html")
    assert report.passed is False
    assert "console error" in (report.error or "")
    assert "Uncaught TypeError" in " ".join(report.log_tail)


def test_html_page_error_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    files = {"index.html": "<html><body>hi</body></html>"}
    _patch_playwright(monkeypatch, console_errors=[], page_errors=["SyntaxError: bad"])
    report = run_build_smoke(files, "html")
    assert report.passed is False
    assert "SyntaxError: bad" in report.log_tail


def test_html_no_entry_file_fails() -> None:
    files = {"styles.css": "body { color: red; }"}  # no .html at all
    report = run_build_smoke(files, "html")
    assert report.passed is False
    assert "no .html entry file" in (report.error or "")


def test_html_playwright_import_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """If Playwright isn't installed, the smoke gate reports a clean failure."""
    files = {"index.html": "<html></html>"}
    import builtins

    real_import = builtins.__import__

    def _block_playwright(name, *args, **kwargs):
        if "playwright" in name:
            raise ImportError("no playwright")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_playwright)
    report = run_build_smoke(files, "html")
    assert report.passed is False
    assert "playwright" in (report.error or "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_playwright(
    monkeypatch: pytest.MonkeyPatch,
    console_errors: List[str],
    page_errors: List[str],
) -> None:
    """Inject a fake sync_playwright context manager that captures handlers.

    The real _smoke_html registers console/pageerror handlers and navigates.
    We fake the page object so that registering the handlers stores them, then
    we invoke them with the canned errors to simulate the page emitting them.
    """

    class _FakePage:
        def __init__(self) -> None:
            self._console_handler: Any = None
            self._pageerror_handler: Any = None

        def on(self, event: str, handler: Any) -> None:
            if event == "console":
                self._console_handler = handler
            elif event == "pageerror":
                self._pageerror_handler = handler

        def goto(self, *_a, **_kw) -> None:
            # Replay the canned errors through the registered handlers.
            for msg in console_errors:
                self._console_handler(_FakeConsole("error", msg))
            for err in page_errors:
                self._pageerror_handler(err)

        def wait_for_timeout(self, _ms: int) -> None:
            pass

    class _FakeConsole:
        def __init__(self, type_: str, text: str) -> None:
            self.type = type_
            self.text = text

    class _FakeBrowser:
        def new_page(self) -> _FakePage:
            return _FakePage()

        def close(self) -> None:
            pass

    class _FakeChromium:
        def launch(self, **_kw) -> _FakeBrowser:
            return _FakeBrowser()

    class _FakePlaywright:
        chromium = _FakeChromium()

    class _FakeCtx:
        def __enter__(self) -> _FakePlaywright:
            return _FakePlaywright()

        def __exit__(self, *_a) -> None:
            pass

    def _fake_sync_playwright() -> _FakeCtx:
        return _FakeCtx()

    # The module imports sync_playwright lazily inside _smoke_html, so patch
    # the name as it would be resolved. We patch sys.modules['playwright.sync_api'].
    import sys
    fake_mod = MagicMock()
    fake_mod.sync_playwright = _fake_sync_playwright
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_mod)
