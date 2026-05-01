"""REST routes for /api/llm — endpoint connectivity + model discovery."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from neme_extractor.llm import LLMUnavailable, discover_models

router = APIRouter(prefix="/api/llm", tags=["llm"])


class DiscoverBody(BaseModel):
    endpoint: str
    # Optional bearer token — empty/missing means "unauthenticated", which is
    # the LMStudio default. Required for OpenAI / OpenRouter / hosted vLLM.
    api_key: str | None = None


@router.post("/discover-models")
async def discover_models_endpoint(body: DiscoverBody) -> dict:
    """Probe a configured OpenAI-compatible server. Doubles as a connectivity
    check for the Settings UI: a 200 with a model list means "endpoint is
    healthy", a 422 means "we tried but it didn't talk back".
    """
    if not body.endpoint or not body.endpoint.strip():
        raise HTTPException(status_code=422, detail="endpoint is required")
    try:
        models = await asyncio.to_thread(
            discover_models, body.endpoint.strip(), body.api_key,
        )
    except LLMUnavailable as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"models": models}
