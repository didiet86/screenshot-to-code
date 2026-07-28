from fastapi import APIRouter
from pydantic import BaseModel

import config

router = APIRouter()


class Capabilities(BaseModel):
    screenshot_preview: bool


@router.get("/api/capabilities", response_model=Capabilities)
async def get_capabilities() -> Capabilities:
    """Backend feature availability for the frontend to reflect in settings."""
    # The screenshot_preview capability is a vision-path feature (spec §6).
    # When VISION_ENABLED is false (the default), we short-circuit to False
    # without importing preview_screenshot — which pulls in playwright and the
    # Chromium backend. This keeps the no-vision deployment dependency-light.
    if not config.VISION_ENABLED:
        return Capabilities(screenshot_preview=False)
    from preview_screenshot import probe_screenshot_preview

    return Capabilities(screenshot_preview=await probe_screenshot_preview())
