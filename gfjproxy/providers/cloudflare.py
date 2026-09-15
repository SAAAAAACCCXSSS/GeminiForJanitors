from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT
from ..http_client import http_client
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..xuiduser import XUID


def _response_text(content: Any) -> str:
    """Extract text from OpenAI-compatible Cloudflare message content."""

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


def _cloudflare_error(response: httpx2.Response) -> tuple[str, str]:
    """Return a readable Cloudflare/OpenAI-compatible error and extra details."""

    message = "Error from Cloudflare Workers AI"
    extras = ""

    try:
        body = response.json()
    except Exception:  # ruff: ignore[BLE001]
        body = None

    if isinstance(body, dict):
        error = body.get("error")

        if isinstance(error, dict):
            if code := error.get("code"):
                message += f" ({code})"
            if error_message := error.get("message"):
                message += f": {error_message}"
            return message, extras

        errors = body.get("errors")
        if isinstance(errors, list) and errors:
            first = errors[0]
            if isinstance(first, dict):
                if code := first.get("code"):
                    message += f" ({code})"
                if error_message := first.get("message"):
                    message += f": {error_message}"

                if len(errors) > 1:
                    extras = str(errors[1:])
                return message, extras

        if detail := body.get("message") or body.get("detail"):
            message += f": {detail}"
            return message, extras

    if response.text:
        extras = response.text

    return message, extras


def cloudflare_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Call Cloudflare Workers AI through its OpenAI-compatible endpoint.

    The API-key payload must be ``ACCOUNT_ID:API_TOKEN``. GeminiForJanitors
    users therefore pass ``cloudflare/ACCOUNT_ID:API_TOKEN`` as their API key.
    Each user supplies their own Cloudflare credentials and consumes only their
    own Workers AI quota.
    """

    if ":" not in api_key:
        return JaiResult(
            400,
            "Invalid Cloudflare credentials.",
            extras=(
                "Use `cloudflare/ACCOUNT_ID:API_TOKEN` as the API key. "
                "Do not put Cloudflare credentials in Render environment variables."
            ),
            metadata=JaiResultMetadata(api_key_valid=False),
        )

    account_id, token = api_key.split(":", maxsplit=1)
    account_id = account_id.strip()
    token = token.strip()

    if not account_id or not token:
        return JaiResult(
            400,
            "Invalid Cloudflare credentials.",
            extras="Both Account ID and API Token are required.",
            metadata=JaiResultMetadata(api_key_valid=False),
        )

    cloudflare_request = {
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
            cloudflare_request["temperature"] = value
        elif key == "max_tokens":
            cloudflare_request["max_tokens"] = value
        elif key == "top_p":
            cloudflare_request["top_p"] = value
        elif key == "frequency_penalty":
            cloudflare_request["frequency_penalty"] = value
        elif key == "repetition_penalty":
            # Workers AI's OpenAI-compatible endpoint exposes presence_penalty.
            cloudflare_request["presence_penalty"] = value

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    endpoint = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id}/ai/v1/chat/completions"
    )

    try:
        cloudflare_response = http_client.post(
            endpoint,
            json=cloudflare_request,
            headers=headers,
            timeout=PROCESS_TIMEOUT,
        )
        cloudflare_response.raise_for_status()
        cloudflare_result = cloudflare_response.json()
    except httpx2.TimeoutException:
        track_stats("cloudflare.time_out")
        return JaiResult(504, "Gateway Timeout")
    except httpx2.HTTPStatusError as e:
        message, extras = _cloudflare_error(e.response)

        if e.response.is_client_error:
            track_stats("cloudflare.failed.client")
        elif e.response.is_server_error:
            track_stats("cloudflare.failed.server")
        else:
            track_stats("cloudflare.failed.unknown")

        return JaiResult(
            e.response.status_code,
            message,
            extras=extras,
            metadata=JaiResultMetadata(
                api_key_valid=e.response.status_code != 401,
            ),
        )
    except Exception as e:  # ruff: ignore[BLE001]
        xlog(user, repr(e))
        track_stats("cloudflare.failed.exception")
        return JaiResult(502, "Unhanded exception from Cloudflare Workers AI.")

    if not isinstance(cloudflare_result, dict):
        xlog(user, f"Unexpected Cloudflare result: {cloudflare_result!r}")
        track_stats("cloudflare.failed.anomalous")
        return JaiResult(502, "Unexpected response from Cloudflare Workers AI.")

    if isinstance(error := cloudflare_result.get("error"), dict) and error:
        message = "Error from Cloudflare Workers AI"
        if error_code := error.get("code"):
            message += f" ({error_code})"
        if error_message := error.get("message"):
            message += f": {error_message}"
        xlog(user, f"Error despite successful status: {cloudflare_result!r}")
        track_stats("cloudflare.failed.anomalous")
        return JaiResult(502, message)

    try:
        content = cloudflare_result["choices"][0]["message"]["content"]
        text = _response_text(content)
    except (KeyError, IndexError, TypeError):
        text = ""

    metadata = JaiResultMetadata()
    if usage := cloudflare_result.get("usage"):
        details = usage.get("completion_tokens_details") or {}
        metadata.token_usage = JaiResultTokenUsage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    if not text:
        xlog(user, f"No result text: {cloudflare_result!r}")
        track_stats("cloudflare.rejected")
        return JaiResult(502, "Response blocked/empty.", metadata=metadata)

    track_stats("cloudflare.succeeded")
    return JaiResult(200, text, metadata=metadata)
