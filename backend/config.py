import os

NUM_VARIANTS = 4
NUM_VARIANTS_VIDEO = 2

# LLM-related
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", None)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)

# Image generation (optional)
REPLICATE_API_KEY = os.environ.get("REPLICATE_API_KEY", None)

# Debugging-related
IS_DEBUG_ENABLED = bool(os.environ.get("IS_DEBUG_ENABLED", False))
DEBUG_DIR = os.environ.get("DEBUG_DIR", "")

# When enabled, every LLM request is written to run_logs/prompt_reports as a
# JSON report viewable at /evals/prompt-reports.
# Hard per-generation spend ceiling; a run that would continue past this is
# aborted. Applies per variant/eval run. Unpriced models are not bounded.
GENERATION_MAX_COST_USD = 3.0

PROMPT_REPORTS_ENABLED = os.environ.get(
    "PROMPT_REPORTS_ENABLED", ""
).strip().lower() in {"1", "true", "yes", "on"}
LOCAL_ASSET_DIR = os.environ.get(
    "LOCAL_ASSET_DIR", os.path.join(os.path.dirname(__file__), "local_assets")
)
# Base URL the backend serves /local-assets from. The live (websocket) path
# infers this per-request; the evals path has no request, so it uses this.
LOCAL_ASSET_BASE_URL = os.environ.get("LOCAL_ASSET_BASE_URL", "http://127.0.0.1:7001")

# Set to True when running in production (on the hosted version)
# Used as a feature flag to enable or disable certain features
IS_PROD = os.environ.get("IS_PROD", False)

# --- No-vision path (spec §3.3) ---------------------------------------------
# Dedicated env vars — do NOT reuse OPENAI_API_KEY / OPENAI_BASE_URL, which
# remain bound to the (deprecated) vision path during the §6 deprecation window.
# Reusing them would collide on a vision rollback (VISION_ENABLED=true).

# LiteLLM gateway (Chat Completions endpoint). Required for the no-vision path.
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", None)
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", None)
# Model name as registered in LiteLLM (e.g. "zai-glm-5.2").
LITELLM_MODEL = os.environ.get("LITELLM_MODEL", None)

# Gate for the legacy vision path (spec §6). Default off — the no-vision path
# is the default; the vision path is reachable only when this is true.
VISION_ENABLED = os.environ.get("VISION_ENABLED", "false").strip().lower() in {
    "1", "true", "yes", "on"
}

# Per-generation $ ceiling for the no-vision path. Conservative default; MUST
# be re-benchmarked from measured data (spec §4.4 / §7.5) — do not trust
# unmeasured. The vision path keeps its own GENERATION_MAX_COST_USD above.
try:
    NO_VISION_BUDGET_USD = float(os.environ.get("NO_VISION_BUDGET_USD", "1.0"))
except ValueError:
    NO_VISION_BUDGET_USD = 1.0

# Shared-secret for the /generate-from-spec endpoint (spec §9.1).
CLONE_DESIGN_API_KEY = os.environ.get("CLONE_DESIGN_API_KEY", None)


def assert_novision_config() -> None:
    """Fail fast with a clear message if required no-vision vars are missing."""
    missing = [
        name
        for name, val in (
            ("LITELLM_BASE_URL", LITELLM_BASE_URL),
            ("LITELLM_API_KEY", LITELLM_API_KEY),
            ("LITELLM_MODEL", LITELLM_MODEL),
        )
        if not val
    ]
    if missing:
        raise RuntimeError(
            "No-vision path requires: " + ", ".join(missing)
            + ". Set these env vars (spec §3.3)."
        )
