from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT
from ..http_client import http_client
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..xuiduser import XUID


def _response_text(content: Any) -> str:
    """Extract text from Mistral chat-completion message content."""

    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return "" if content is None else str(content)

    parts: list[str] = []

    for chunk in content:
        if isinstance(chunk, str):
            parts.append(chunk)
            continue

        if not isinstance(chunk, dict):
            continue

        text = chunk.get("text")
        if isinstance(text, str):
            parts.append(text)

    return "".join(parts)


def mistral_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Wrapper around Mistral's OpenAI-compatible Chat Completions API.

    The user parameter is only used for logging.
    """

    mistral_request = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "content": message.content,
                "role": message.role,
            }
            for message in messages
        ],
    }

    for key, value in (settings or {}).items():
        if key == "temperature":
            mistral_request["temperature"] = value
        elif key == "max_tokens":
            mistral_request["max_tokens"] = value
        elif key == "top_p":
            mistral_request["top_p"] = value
        elif key == "frequency_penalty":
            mistral_request["frequency_penalty"] = value
        elif key == "repetition_penalty":
            # Tavo/Janitor call this repetition_penalty. Mistral exposes
            # presence_penalty on its OpenAI-compatible endpoint.
            mistral_request["presence_penalty"] = value

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        mistral_response = http_client.post(
            "https://api.mistral.ai/v1/chat/completions",
            json=mistral_request,
            headers=headers,
            timeout=PROCESS_TIMEOUT,
        )
        mistral_response.raise_for_status()
        mistral_result = mistral_response.json()
    except httpx2.TimeoutException:
        track_stats("mistral.time_out")
        return JaiResult(504, "Gateway Timeout")
    except httpx2.HTTPStatusError as e:
        message = "Error from Mistral"
        extras = ""

        try:
            error_body = e.response.json()
        except Exception:  # ruff: ignore[BLE001]
            error_body = {}

        error = (
            error_body.get("error", error_body)
            if isinstance(error_body, dict)
            else {}
        )

        if isinstance(error, dict):
            if error_code := error.get("code"):
                message += f" ({error_code})"
            if error_message := error.get("message"):
                message += f": {error_message}"
        elif e.response.text:
            extras = e.response.text

        if e.response.is_client_error:
            track_stats("mistral.failed.client")
        elif e.response.is_server_error:
            track_stats("mistral.failed.server")
        else:
            track_stats("mistral.failed.unknown")

        metadata = JaiResultMetadata(
            api_key_valid=e.response.status_code not in (401, 403),
        )

        return JaiResult(
            e.response.status_code,
            message,
            extras=extras,
            metadata=metadata,
        )
    except Exception as e:  # ruff: ignore[BLE001]
        xlog(user, repr(e))
        track_stats("mistral.failed.exception")
        return JaiResult(502, "Unhanded exception from Mistral.")

    if isinstance(error := mistral_result.get("error"), dict) and error:
        message = "Error from Mistral"
        if error_code := error.get("code"):
            message += f" ({error_code})"
        if error_message := error.get("message"):
            message += f": {error_message}"
        xlog(user, f"Error despite successful status: {mistral_result!r}")
        track_stats("mistral.failed.anomalous")
        return JaiResult(502, message)

    try:
        content = mistral_result["choices"][0]["message"]["content"]
        text = _response_text(content)
    except (KeyError, IndexError, TypeError):
        text = ""

    metadata = JaiResultMetadata()
    if usage := mistral_result.get("usage"):
        metadata.token_usage = JaiResultTokenUsage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    if not text:
        xlog(user, f"No result text: {mistral_result!r}")
        track_stats("mistral.rejected")
        return JaiResult(502, "Response blocked/empty.", metadata=metadata)

    track_stats("mistral.succeeded")
    return JaiResult(200, text, metadata=metadata)
