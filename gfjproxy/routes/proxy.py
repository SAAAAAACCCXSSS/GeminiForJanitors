from traceback import print_exception

from flask import Blueprint, abort, request

from ..cooldown import get_cooldown
from ..handlers import PROVIDER_FUNCS, handle_chat_message, handle_proxy_test
from ..logging import xlog, xlogtime
from ..models import JaiRequest
from ..providers.cloudflare import cloudflare_generate_content
from ..providers.mistral import mistral_generate_content
from ..storage import storage
from ..utils import ResponseHelper, comma_split, is_proxy_test
from ..xuid_secret import xuid_secret
from ..xuiduser import XUID, UserSettings

proxy = Blueprint("proxy", __name__)

PROVIDER_FUNCS["cloudflare"] = cloudflare_generate_content
PROVIDER_FUNCS["mistral"] = mistral_generate_content


def _debug_raw_user_messages(request_json: dict) -> None:
    """Temporary safe diagnostic for Tavo/Janitor request shapes.

    Does NOT log Authorization/API keys. It prints only short previews of
    user-role message content so we can see whether Tavo changed the payload.
    """

    messages = request_json.get("messages")

    if not isinstance(messages, list):
        print("[TAVO DEBUG] request has no messages list", flush=True)
        return

    print(
        f"[TAVO DEBUG] raw message roles="
        f"{[m.get('role') if isinstance(m, dict) else '?' for m in messages]}",
        flush=True,
    )

    user_messages = [
        (i, m)
        for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") == "user"
    ]

    for i, message in user_messages[-4:]:
        content = message.get("content")
        content_type = type(content).__name__

        if isinstance(content, str):
            preview = content

        elif isinstance(content, list):
            text_parts: list[str] = []

            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)

            preview = "\n".join(text_parts)

        else:
            preview = repr(content)

        preview = preview.replace("\r", "\\r").replace("\n", "\\n")
        preview = preview[:700]

        print(
            f"[TAVO DEBUG] user[{i}] "
            f"content_type={content_type} "
            f"text={preview!r}",
            flush=True,
        )


# JanitorAI routes
@proxy.route("/", methods=["POST"])
@proxy.route("/chat/completions", methods=["POST"])
@proxy.route("/quiet/", methods=["POST"])
@proxy.route("/quiet/chat/completions", methods=["POST"])

# OpenAI-compatible routes for Tavo / SillyTavern / other frontends
@proxy.route("/v1/chat/completions", methods=["POST"])
@proxy.route("/v1/quiet/chat/completions", methods=["POST"])
def handle():
    assert storage is not None

    request_json = request.get_json(silent=True)

    if not request_json:
        abort(400, "Bad Request. Missing or invalid JSON.")
        return

    # Temporary diagnostics. No API key/header is printed.
    _debug_raw_user_messages(request_json)

    request_path = request.path

    jai_req = JaiRequest.parse(request_json)

    jai_req.quiet = "/quiet/" in request_path

    proxy_test = is_proxy_test(request_json)

    response = ResponseHelper(
        use_stream=jai_req.stream,
        wrap_errors=proxy_test,
    )

    request_auth = request.headers.get("authorization", "").split(" ", maxsplit=1)

    if len(request_auth) != 2 or request_auth[0].lower() != "bearer":
        return response.build_error(
            "Unauthorized. API key required.",
            401,
        )

    api_keys = comma_split(request_auth[1])

    xuid = XUID(
        api_keys[0],
        xuid_secret,
    )

    user = UserSettings(
        storage,
        xuid,
    )

    if (
        (seconds := user.last_seen())
        and (cooldown := get_cooldown())
        and (delay := cooldown - seconds) > 0
    ):
        xlog(
            user,
            f"User told to wait {delay} seconds",
        )

        return response.build_error(
            f"Please wait {delay} seconds.",
            429,
        )

    rcounter = user.get_rcounter()

    api_key_index = rcounter % len(api_keys)

    user.inc_rcounter()

    selected_api_key = api_keys[api_key_index]

    if set(jai_req.models) == {"mistral"} and "/" not in selected_api_key:
        selected_api_key = f"mistral/{selected_api_key}"

    jai_req.api_key = selected_api_key
    jai_req.api_key_index = api_key_index
    jai_req.api_key_count = len(api_keys)

    log_details = [
        f"User {user.last_seen_msg()}",
        f"Request #{user.get_rcounter()}",
    ]

    if len(api_keys) > 1:
        log_details.append(
            f"Key {api_key_index + 1}/{len(api_keys)}"
        )

    ref_time = xlogtime(
        user,
        (
            f"Processing "
            f"{'stream ' if jai_req.stream else ''}"
            f"{request_path} "
            f"({', '.join(log_details)})"
        ),
    )

    try:
        if not jai_req.models:
            response.add_error(
                "Please specify a model.",
                400,
            )

        elif unknown := jai_req.models.get("unknown"):
            response.add_error(
                (
                    f"Unknown model(s): {unknown}\n"
                    "Make sure to use OpenRouter syntax "
                    "`provider/model`.\n"
                    "Examples: "
                    "`google/gemini-2.5-flash`, "
                    "`cerebras/llama3.1-8b`, "
                    "`deepseek/deepseek-chat`, etc."
                ),
                400,
            )

        elif proxy_test:
            response = handle_proxy_test(
                user,
                jai_req,
                response,
            )

        else:
            response = handle_chat_message(
                user,
                jai_req,
                response,
            )

    except Exception as e:
        response.add_error(
            "Internal Proxy Error",
            500,
        )

        print_exception(e)

    if 200 <= response.status <= 299:
        xlogtime(
            user,
            "Processing succeeded",
            ref_time,
        )

        if not proxy_test and (announcement := storage.announcement):
            response.add_proxy_message(
                f"***\n{announcement}\n***"
            )

    else:
        messages = response.message.split("\n")

        xlogtime(
            user,
            f"Processing failed: {messages[0]}",
            ref_time,
        )

        for message in messages[1:]:
            xlog(
                user,
                f"> {message}",
            )

    if user.valid:
        user.save()
    else:
        xlog(
            user,
            "Invalid user not saved",
        )

    return response.build()
