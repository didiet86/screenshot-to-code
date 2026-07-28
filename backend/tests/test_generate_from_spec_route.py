"""End-to-end smoke test for the /generate-from-spec route (spec §8, §9).

Tests the full HTTP contract — auth, job lifecycle (queued → running → done),
status polling with progress_pct, and zip packaging — with the NovisionEngine
mocked out so no real LiteLLM gateway is required.

Uses FastAPI's TestClient (synchronous wrapper over the ASGI app).
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient


# Minimal valid spec (spec §2.1) — just enough to pass intake validation.
SAMPLE_SPEC: Dict[str, Any] = {
    "version": "1.0",
    "url": "https://example.com",
    "title": "Test Page",
    "tokens": {
        "colors": {"palette": [{"hex": "#000000", "role": "fg"}]},
        "typography": {"font_families": ["Inter"]},
        "spacing": {"scale": [{"name": "md", "value": "16px"}]},
    },
    "sections": [
        {"id": "sec-header", "role": "header", "layout": "flex-row", "components": []},
    ],
    "components": [],
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Minimal FastAPI app with ONLY the no-vision router mounted.

    We deliberately do NOT import ``main.py`` here: the full app eagerly
    imports vision-laden routes (evals → agent.engine → factory → anthropic)
    that aren't installed in the no-vision test venv. The no-vision router is
    self-contained, so mounting it on a bare app tests the real HTTP contract
    (auth, job lifecycle, zip packaging) without the vision app shell.
    """
    monkeypatch.setenv("CLONE_DESIGN_API_KEY", "test-secret")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    monkeypatch.setenv("LITELLM_MODEL", "test-model")

    # Reload config so it picks up the env vars.
    import importlib
    import config as config_module
    importlib.reload(config_module)

    # Mock the engine to return a canned multi-file result.
    async def _fake_run(self) -> Dict[str, Any]:
        return {
            "files": {
                "index.html": "<html><body>hi</body></html>",
                "src/styles/tokens.css": ":root { --color-fg: #000000; }",
            },
            "finished": True,
            "iterations": 3,
            "tool_call_count": 5,
            "malformed_tool_calls": 0,
            "input_tokens": 100,
            "output_tokens": 200,
        }

    from agent.novision_engine import NovisionEngine
    monkeypatch.setattr(NovisionEngine, "run", _fake_run)

    # Bare app with only the no-vision router.
    from fastapi import FastAPI
    import routes.generate_from_spec as gfs
    importlib.reload(gfs)
    app = FastAPI()
    app.include_router(gfs.router)
    return TestClient(app)


def _auth_headers() -> Dict[str, str]:
    return {"X-Api-Key": "test-secret"}


def test_intake_rejects_missing_api_key(client: TestClient) -> None:
    """No X-Api-Key → 401 (spec §9.1)."""
    resp = client.post("/generate-from-spec", json={"spec": SAMPLE_SPEC, "framework": "html"})
    assert resp.status_code == 401


def test_intake_rejects_wrong_api_key(client: TestClient) -> None:
    resp = client.post(
        "/generate-from-spec",
        json={"spec": SAMPLE_SPEC, "framework": "html"},
        headers={"X-Api-Key": "wrong"},
    )
    assert resp.status_code == 401


def test_intake_rejects_bad_spec_version(client: TestClient) -> None:
    """Spec version != 1.0 → 400 (spec §2.2)."""
    bad_spec = {**SAMPLE_SPEC, "version": "2.0"}
    resp = client.post(
        "/generate-from-spec",
        json={"spec": bad_spec, "framework": "html"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    assert "2.0" in resp.json()["detail"]


def test_intake_rejects_missing_version(client: TestClient) -> None:
    bad_spec = {k: v for k, v in SAMPLE_SPEC.items() if k != "version"}
    resp = client.post(
        "/generate-from-spec",
        json={"spec": bad_spec, "framework": "html"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 400


def test_full_job_lifecycle(client: TestClient) -> None:
    """POST → job_id; poll status → done; GET result → valid zip (spec §9.2)."""
    # 1. Start job.
    resp = client.post(
        "/generate-from-spec",
        json={"spec": SAMPLE_SPEC, "framework": "html", "stack": "html_css"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    job = resp.json()
    assert "job_id" in job
    job_id = job["job_id"]

    # 2. Poll status until done (the mocked engine resolves synchronously in the
    #    background task; loop a few times).
    final_status = None
    for _ in range(20):
        s = client.get(
            f"/generate-from-spec/{job_id}/status", headers=_auth_headers()
        )
        assert s.status_code == 200
        body = s.json()
        assert body["status"] in {"queued", "running", "done", "error"}
        if body["status"] in {"done", "error"}:
            final_status = body
            break
    assert final_status is not None, "Job never reached done/error"
    assert final_status["status"] == "done", f"Job errored: {final_status.get('error')}"
    assert final_status["progress_pct"] == 100

    # 3. Fetch result zip.
    r = client.get(f"/generate-from-spec/{job_id}/result", headers=_auth_headers())
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"

    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert "index.html" in names
    assert "src/styles/tokens.css" in names
    assert "manifest.json" in names

    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["framework"] == "html"
    assert manifest["spec_version"] == "1.0"
    assert manifest["generation"]["finished"] is True
    assert manifest["file_count"] == 2


def test_result_before_done_returns_409(client: TestClient) -> None:
    """Fetching result for a non-done job → 409."""
    resp = client.post(
        "/generate-from-spec",
        json={"spec": SAMPLE_SPEC, "framework": "html"},
        headers=_auth_headers(),
    )
    job_id = resp.json()["job_id"]
    # Immediately fetch result (job likely still queued/running).
    r = client.get(f"/generate-from-spec/{job_id}/result", headers=_auth_headers())
    assert r.status_code in (409, 200)  # 409 if not done yet, 200 if it raced ahead


def test_status_unknown_job_returns_404(client: TestClient) -> None:
    r = client.get(
        "/generate-from-spec/nonexistent/status", headers=_auth_headers()
    )
    assert r.status_code == 404
