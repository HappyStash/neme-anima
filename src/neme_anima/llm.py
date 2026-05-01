"""Image-description via an OpenAI-compatible chat-completions endpoint.

Targets LMStudio first (``http://localhost:1234``) — its ``/v1/models`` and
``/v1/chat/completions`` endpoints follow the OpenAI shape closely, so any
other compatible server (Ollama with ``--openai``, vLLM, etc.) works without
changes. Vision is delivered as a base64 ``image_url`` data URL inline with
the user message, which is what LMStudio + GGUF VLMs accept.
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx


DEFAULT_PROMPT = (
    "Describe this image in 1-2 sentences for a LoRA training caption. "
    "Focus on the subject's pose, clothing, expression, background, lighting, "
    "and any distinctive details. Be concise, factual, and avoid speculating "
    "about names, intent, or off-camera context."
)
DEFAULT_ENDPOINT = "http://localhost:1234"

# Connect quickly so the UI doesn't hang on a typo'd endpoint, but allow
# generous time for the model itself to think on a long-ish vision prompt.
_MODELS_TIMEOUT = httpx.Timeout(connect=3.0, read=10.0, write=5.0, pool=5.0)
_DESCRIBE_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=10.0)


class LLMUnavailable(RuntimeError):
    """Raised when the configured endpoint can't be reached or returned an error."""


def _normalize_endpoint(endpoint: str) -> str:
    return endpoint.rstrip("/")


def _auth_headers(api_key: str | None) -> dict[str, str]:
    """Return Bearer-auth headers when an API key is set, else an empty dict.

    LMStudio (the default target) doesn't gate either endpoint — sending an
    Authorization header is harmless but emitting one only when needed keeps
    server-side logs clean and avoids confusing intermediaries.
    """
    if api_key and api_key.strip():
        return {"Authorization": f"Bearer {api_key.strip()}"}
    return {}


def discover_models(endpoint: str, api_key: str | None = None) -> list[str]:
    """Return the model IDs the endpoint exposes via ``GET /v1/models``."""
    url = f"{_normalize_endpoint(endpoint)}/v1/models"
    try:
        resp = httpx.get(url, timeout=_MODELS_TIMEOUT, headers=_auth_headers(api_key))
    except httpx.HTTPError as exc:
        raise LLMUnavailable(f"could not reach {url}: {exc}") from exc
    if resp.status_code != 200:
        raise LLMUnavailable(
            f"{url} returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    try:
        data = resp.json()
    except ValueError as exc:
        raise LLMUnavailable(f"non-JSON response from {url}: {exc}") from exc
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise LLMUnavailable(f"unexpected schema from {url}: missing 'data' list")
    out: list[str] = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("id"), str):
            out.append(it["id"])
    return sorted(out)


def _image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower().lstrip(".") or "png"
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def describe_image(
    *,
    endpoint: str,
    model: str,
    image_path: Path,
    prompt: str = DEFAULT_PROMPT,
    danbooru_tags: str | None = None,
    api_key: str | None = None,
) -> str:
    """Send the image to ``/v1/chat/completions`` and return the description text.

    ``danbooru_tags`` is passed as additional grounding context so the VLM
    doesn't contradict the tagger's labels — useful when the LoRA pipeline
    cares about both lines staying coherent.
    """
    url = f"{_normalize_endpoint(endpoint)}/v1/chat/completions"
    user_text = prompt
    if danbooru_tags:
        user_text = (
            f"{prompt}\n\nReference tags from a tagger (use as hints, not as a "
            f"verbatim copy): {danbooru_tags}"
        )
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url",
                 "image_url": {"url": _image_to_data_url(image_path)}},
            ],
        }],
        "temperature": 0.2,
        "max_tokens": 200,
    }
    try:
        resp = httpx.post(
            url, json=body, timeout=_DESCRIBE_TIMEOUT,
            headers=_auth_headers(api_key),
        )
    except httpx.HTTPError as exc:
        raise LLMUnavailable(f"could not reach {url}: {exc}") from exc
    if resp.status_code != 200:
        raise LLMUnavailable(
            f"{url} returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    try:
        data = resp.json()
    except ValueError as exc:
        raise LLMUnavailable(f"non-JSON response from {url}: {exc}") from exc
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMUnavailable(f"unexpected response shape from {url}: {data!r}") from exc
    if not isinstance(text, str):
        raise LLMUnavailable(f"non-string content from {url}: {text!r}")
    return _clean_description(text)


def _clean_description(text: str) -> str:
    """Collapse to a single line — LoRA caption sidecars are line-delimited."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return " ".join(lines)
