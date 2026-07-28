# Load environment variables first
from dotenv import load_dotenv

load_dotenv()


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import IS_DEBUG_ENABLED, VISION_ENABLED

# --- No-vision routes (always available) ------------------------------------
# These imports must be vision-free (spec §3.3 rule 1). They pull in neither
# playwright, websockets, nor the vision tool runtime.
from routes import (
    capabilities,
    generate_from_spec,
    home,
    evals,
    export,
    design_systems,
    prompt_reports,
    agent_runs,
    eval_sets,
)

app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)


@app.on_event("startup")
async def log_debug_mode() -> None:
    debug_status = "ENABLED" if IS_DEBUG_ENABLED else "DISABLED"
    mode = "VISION" if VISION_ENABLED else "NO-VISION"
    print(f"Backend startup complete. Debug mode is {debug_status}. Mode: {mode}.")


# --- Vision routes (DEPRECATED, spec §6) ------------------------------------
# Only imported/registered when VISION_ENABLED=true. The no-vision deployment
# (default) never imports these, so playwright/websockets/uploaded_assets are
# not required dependencies for a no-vision instance.
if VISION_ENABLED:
    from routes import generate_code, screenshot
    from uploaded_assets import configure_uploaded_asset_routes

    configure_uploaded_asset_routes(app)


@app.on_event("startup")
async def probe_screenshot_preview_on_startup() -> None:
    # DEPRECATED: vision path startup probe (spec §6). Only runs when
    # VISION_ENABLED=true. The no-vision path (default) does not need
    # headless Chromium, so we skip the probe — and the playwright import —
    # entirely, keeping the no-vision deployment dependency-light.
    if not VISION_ENABLED:
        return
    # Detect (and warm up) headless Chromium so the screenshot_preview tool is
    # only offered when it can actually run. Logs the outcome.
    from preview_screenshot import probe_screenshot_preview

    await probe_screenshot_preview()

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routes
if VISION_ENABLED:
    app.include_router(generate_code.router)
    app.include_router(screenshot.router)
app.include_router(generate_from_spec.router)  # no-vision path (spec §8)
app.include_router(home.router)
app.include_router(capabilities.router)
app.include_router(evals.router)
app.include_router(export.router)
app.include_router(design_systems.router)
app.include_router(prompt_reports.router)
app.include_router(agent_runs.router)
app.include_router(eval_sets.router)
