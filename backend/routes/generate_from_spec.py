"""HTTP route for the no-vision spec-driven codegen path (spec §8, §9).

Endpoints:
  POST /generate-from-spec            — start a generation job, returns job_id
  GET  /generate-from-spec/{id}/status — poll: {status, progress_pct}
  GET  /generate-from-spec/{id}/result — fetch the finished project zip

Auth: ``X-Api-Key`` header checked against ``CLONE_DESIGN_API_KEY`` (spec §9.1).

Jobs run in background tasks; status is tracked in an in-memory dict. This is
sufficient for a single-instance deployment. For multi-instance, swap the
store for Redis (out of scope here).

Vision-free import graph: imports only the no-vision engine + config + stdlib.
Does NOT import ``routes.generate_code`` (which pulls in the vision WebSocket
path, image handling, etc.).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

import config
from agent.novision_engine import NovisionEngine
from codegen.build_smoke import run_build_smoke
from codegen.project_assembler import assemble_project
from codegen.quality_checks import run_quality_checks

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GenerateFromSpecRequest(BaseModel):
    spec: Dict[str, Any] = Field(..., description="design-spec.json (spec §2)")
    framework: str = Field("next", description="html | next | nuxt | astro")
    stack: str = Field("tailwind", description="tailwind | html_css")
    assets_inline: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional {filename: base64-data-url} of asset bytes (spec §9.3).",
    )


class JobAccepted(BaseModel):
    job_id: str
    status_url: str
    result_url: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | done | error
    progress_pct: int
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

class _Job:
    def __init__(self, job_id: str, request: GenerateFromSpecRequest) -> None:
        self.job_id = job_id
        self.request = request
        self.status: str = "queued"
        self.progress_pct: int = 0
        self.error: Optional[str] = None
        self.result_files: Dict[str, str] = {}
        self.result_meta: Dict[str, Any] = {}
        self.created_at: float = time.time()


_jobs: Dict[str, _Job] = {}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _check_api_key(x_api_key: Optional[str]) -> None:
    expected = config.CLONE_DESIGN_API_KEY
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server misconfigured: CLONE_DESIGN_API_KEY not set.",
        )
    if not x_api_key or x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Api-Key.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/generate-from-spec", response_model=JobAccepted)
async def generate_from_spec(
    request: GenerateFromSpecRequest,
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
) -> JobAccepted:
    _check_api_key(x_api_key)
    config.assert_novision_config()

    # Validate spec version (spec §2.2).
    spec_version = str(request.spec.get("version", ""))
    if not spec_version:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="spec.version is required (spec §2.2).",
        )
    if spec_version != "1.0":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported spec version '{spec_version}'. Expected '1.0'.",
        )

    job_id = f"job-{int(time.time() * 1000)}"
    job = _Job(job_id, request)
    _jobs[job_id] = job

    asyncio.create_task(_run_job(job))
    return JobAccepted(
        job_id=job_id,
        status_url=f"/generate-from-spec/{job_id}/status",
        result_url=f"/generate-from-spec/{job_id}/result",
    )


@router.get("/generate-from-spec/{job_id}/status", response_model=JobStatus)
async def get_status(
    job_id: str,
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
) -> JobStatus:
    _check_api_key(x_api_key)
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatus(
        job_id=job_id,
        status=job.status,
        progress_pct=job.progress_pct,
        error=job.error,
    )


@router.get("/generate-from-spec/{job_id}/result")
async def get_result(
    job_id: str,
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
) -> Response:
    _check_api_key(x_api_key)
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != "done":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is {job.status}, not done.",
        )

    zip_bytes = assemble_project(
        files=job.result_files,
        spec=job.request.spec,
        framework=job.request.framework,
        stack=job.request.stack,
        generation_meta=job.result_meta,
    )
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'},
    )


# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------

async def _run_job(job: _Job) -> None:
    job.status = "running"
    job.progress_pct = 5
    try:
        engine = NovisionEngine(
            spec=job.request.spec,
            framework=job.request.framework,
            stack=job.request.stack,
            model=config.LITELLM_MODEL,  # type: ignore[arg-type]
            base_url=config.LITELLM_BASE_URL,  # type: ignore[arg-type]
            api_key=config.LITELLM_API_KEY,  # type: ignore[arg-type]
            budget_usd=config.NO_VISION_BUDGET_USD,
            on_event=lambda e: _on_event(job, e),
        )
        job.progress_pct = 10
        result = await engine.run()
        job.result_files = result.get("files", {})

        # --- Quality gates (spec §7) ---
        # Run after generation, before marking done. A gate failure does NOT
        # discard the output — it is recorded in the manifest so the consumer
        # (the app) can decide whether to accept or fall back to Stage 12.
        quality_report = run_quality_checks(job.result_files, job.request.spec)
        smoke_report = run_build_smoke(job.result_files, job.request.framework)

        job.result_meta = {
            "framework": job.request.framework,
            "stack": job.request.stack,
            "spec_version": str(job.request.spec.get("version")),
            "finished": result.get("finished"),
            "iterations": result.get("iterations"),
            "tool_call_count": result.get("tool_call_count"),
            "malformed_tool_calls": result.get("malformed_tool_calls"),
            "section_count": len(job.request.spec.get("sections", [])),
            "file_count": len(job.result_files),
            "quality": quality_report.to_dict(),
            "smoke": smoke_report.to_dict(),
        }
        if not quality_report.passed:
            logger.warning(
                "No-vision job %s failed quality gates: %s",
                job.job_id,
                quality_report.violations,
            )
        if not smoke_report.passed:
            logger.warning(
                "No-vision job %s failed build smoke: %s",
                job.job_id,
                smoke_report.error,
            )
        job.progress_pct = 100
        job.status = "done"
    except Exception as exc:  # noqa: BLE001
        logger.exception("No-vision job %s failed", job.job_id)
        job.error = str(exc)
        job.status = "error"


async def _on_event(job: _Job, event: Dict[str, Any]) -> None:
    """Map agent events to coarse progress (spec §9.2)."""
    etype = event.get("type")
    if etype == "finish":
        job.progress_pct = 95
    elif etype == "tool_result":
        # Nudge progress upward with each tool result, capped at 90.
        job.progress_pct = min(90, max(10, job.progress_pct + 3))


# ---------------------------------------------------------------------------
# Zip packaging is handled by codegen.project_assembler.assemble_project (spec §5).
