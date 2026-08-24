import re
from random import randint
from typing import Any, cast

from ._globals import BANNER, BANNER_VERSION
from .commands import CommandError, CommandExit
from .logging import xlog
from .models import JaiMessage, JaiRequest, JaiResult, JaiResultMetadata
from .prefill import apply_prefill, clear_prefill
from .providers.cerebras import cerebras_generate_content
from .providers.deepseek import deepseek_generate_content
from .providers.gemini import gemini_generate_content
from .providers.gemini_cli import gemini_cli_generate_content
from .providers.nvidia import nvidia_generate_content
from .providers.openrouter import openrouter_generate_content
from .providers.proxy import proxy_generate_content
from .providers.z_ai import z_ai_generate_content
from .statistics import track_stats
from .utils import ResponseHelper
from .xuiduser import XUID, UserSettings

################################################################################

API_KEY_PREFIXES = {
    "AIza": "google",  # Standard API keys
    "AQ.": "google",  # Authorization keys
    "csk-": "cerebras",
    "nvapi-": "nvidia",
    "sk-ant-": "anthropic",
    "sk-or-v1-": "openrouter",
    "sk-proj-": "openai",
    "gfjproxy.gemini_cli.": "gemini_cli",
}

PROVIDER_FUNCS = {
    "cerebras": cerebras_generate_content,
    "deepseek": deepseek_generate_content,
    "gemini_cli": gemini_cli_generate_content,
    "google": gemini_generate_content,
    "nvidia": nvidia_generate_content,
    "openrouter": openrouter_generate_content,
    "proxy": proxy_generate_content,
    "z_ai": z_ai_generate_content,
}


################################################################################
# Multimodal helpers
################################################################################


def _message_text(content: Any) -> str:
    """
    Return only the textual portion of a message.

    Normal JanitorAI messages use:
        content: "text"

    OpenAI/Tavo multimodal messages may use:
        content: [
            {"type": "text", "text": "..."},
            {
                "type": "image_url",
                "image_url": {"url": "..."}
            }
        ]

    This helper is used only for internal text inspection.
    It does NOT remove images from the original message.
    """

    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        if content is None:
            return ""
        return str(content)

    text_parts: list[str] = []

    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")

        if block_type in ("text", "input_text"):
            text = block.get("text")

            if isinstance(text, str):
                text_parts.append(text)

            continue

        # Be tolerant of OpenAI-compatible clients that omit "type"
        # but still provide a text property.
        text = block.get("text")

        if isinstance(text, str):
            text_parts.append(text)

    return "\n".join(text_parts)


def _replace_message_text(
    content: Any,
    old: str,
    new: str,
) -> Any:
    """
    Replace text inside either a plain string or multimodal content.

    Image blocks and other non-text blocks are preserved unchanged.
    """

    if isinstance(content, str):
        return content.replace(old, new)

    if not isinstance(content, list):
        return content

    result: list[Any] = []

    for block in content:
        if not isinstance(block, dict):
            result.append(block)
            continue

        new_block = dict(block)

        text = new_block.get("text")

        if isinstance(text, str):
            new_block["text"] = text.replace(old, new)

        result.append(new_block)

    return result


def _has_multimodal_content(messages: list[JaiMessage]) -> bool:
    """Return True if at least one message contains structured content."""

    return any(isinstance(message.content, list) for message in messages)


################################################################################


def _resolve_provider(api_key: str) -> tuple[str | None, str]:
    """Resolves which provider an API key belongs to.

    Returns:
        provider (str | None): The provider's name if any.
        api_key (str): The cleaned up API key.
    """

    api_key_split = api_key.split("/", maxsplit=1)

    if len(api_key_split) == 2:  # "provider/api_key" syntax
        return api_key_split[0].lower(), api_key_split[1]

    # The API key is plain and needs to be pattern matched
    for prefix, provider in API_KEY_PREFIXES.items():
        if api_key.startswith(prefix):
            return provider, api_key

    return None, api_key


def _handle_request(
    user: XUID,
    api_key: str,
    models: dict[str, str],
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Dispatch a JaiRequest request to the appropriate provider given the API key."""

    provider_name, api_key = _resolve_provider(api_key)

    if not provider_name:
        return JaiResult(
            400,
            "The proxy couldn't recognize an API key.",
            extras=(
                f"Your API key `{api_key}` didn't match any of the proxy's prefixes.\n"
                "You should specify the provider at the start of your API key. For example:\n"
                "- If the key is for Cerebras, add `cerebras/` at the start of it.\n"
                "- If the key is for DeepSeek, add `deepseek/` at the start of it.\n"
                "- If the key is for Google AI or Vertex AI, add `google/` at the start of it.\n"
                "- If the key is for Nvidia NIM, add `nvidia/` at the start of it.\n"
                "- If the key is for Z.AI, add `z_ai/` at the start of it.\n"
                "- If the key is for OpenRouter, add `openrouter/` at the start of it.\n"
            ),
            metadata=JaiResultMetadata(api_key_valid=False),
        )

    provider_func = PROVIDER_FUNCS.get(provider_name)

    if not provider_func:
        return JaiResult(
            500,
            f"You have a `{provider_name}` API key but this proxy does not support it.",
        )

    model = models.get(provider_name)

    if not model:
        extras = (
            f"You have a `{provider_name}` API key but you didn't specify a model for it.\n"
            "Make sure to use OpenRouter model syntax `provider/model`.\n"
            "Examples: `google/gemini-2.5-flash`, "
            "`cerebras/llama3.1-8b`, `deepseek/deepseek-chat`, etc."
        )

        if provider_name in ("openrouter", "nvidia"):
            extras += (
                "\n**Note For OpenRouter and Nvidia NIM API keys**:"
                " use an extended model name:"
                " `openrouter/anthropic/claude-3.5-sonnet`,"
                " `nvidia/deepseek-ai/deepseek-v4-pro`, etc."
            )

        return JaiResult(
            400,
            f"Missing model for {provider_name}",
            extras=extras,
        )

    xlog(user, f"Using {provider_name}/{model}")

    return provider_func(
        user,
        api_key,
        model,
        messages,
        settings,
    )


################################################################################


PERSONA_REGEX = re.compile(r"</([^<>]+?)'s Persona>")


def parse_user_persona_names(
    user: UserSettings,
    jai_req: JaiRequest,
) -> tuple[str, str]:
    # JanitorAI usually sends at least four messages with roles:
    # system, user, assistant, user.
    #
    # Tavo/OpenAI-compatible clients may send a different layout,
    # so all text inspection here must tolerate multimodal content.

    if len(jai_req.messages) < 4:
        return "User", "Narrator"

    user_name: str | None = None

    first_user_message = next(
        (
            m
            for m in jai_req.messages
            if m.role == "user"
            and len(_message_text(m.content)) > 1
        ),
        None,
    )

    if first_user_message is not None:
        first_user_text = _message_text(
            first_user_message.content,
        )

        user_name_index = first_user_text.find(": ")

        if user_name_index > 0:
            user_name = first_user_text[
                :user_name_index
            ].strip()

            xlog(
                user,
                f"Parsed user name: {user_name!r}",
            )

    if not user_name:
        xlog(user, "User name not parsed")
        user_name = "User"

    persona_name: str | None = None

    system_message = jai_req.messages[0]
    system_text = _message_text(
        system_message.content,
    )

    if (
        system_message.role == "system"
        and (
            persona_match := PERSONA_REGEX.search(
                system_text,
            )
        )
    ):
        persona_name = str(
            persona_match.group(1)
        ).strip()

        xlog(
            user,
            f"Parsed persona name: {persona_name!r}",
        )

    if not persona_name:
        xlog(user, "Persona name not parsed")
        persona_name = "Narrator"

    return user_name, persona_name


################################################################################


def handle_proxy_test(
    user: UserSettings,
    jai_req: JaiRequest,
    response: ResponseHelper,
) -> ResponseHelper:
    """Proxy test handler.

    The sole purpose of this is to test out the user's API key and model.
    """

    # Pass no settings. Defaults should allow for a successful proxy test.
    result = _handle_request(
        user.xuid,
        jai_req.api_key,
        jai_req.models,
        jai_req.messages,
    )

    user.valid = result.metadata.api_key_valid

    if not result:
        track_stats("r.test.failed")

        extra = ""

        if result.extras:
            extra = "\n(Send a chat message to get the full error)"

        return response.add_error(
            result.error + extra,
            result.status,
        )

    track_stats("r.test.succeeded")

    return response.add_message(
        "TEST"
    )


def handle_chat_message(
    user: UserSettings,
    jai_req: JaiRequest,
    response: ResponseHelper,
) -> ResponseHelper:
    """Chat message handler.

    This handles when the user sends a simple chat message to the bot.
    """

    xlog(
        user,
        f"Request has {len(jai_req.messages)} message(s) with role(s): "
        + "".join(
            m.role[0] if m.role else "?"
            for m in jai_req.messages
        ),
    )

    multimodal_request = _has_multimodal_content(
        jai_req.messages,
    )

    if multimodal_request:
        xlog(
            user,
            "Multimodal/OpenAI-style content detected",
        )

    user_name, persona_name = parse_user_persona_names(
        user,
        jai_req,
    )

    last_user_message = jai_req.messages[-1]

    if jai_req.messages[-1].role == "assistant":
        xlog(user, "User set prefill detected")

        if len(jai_req.messages) >= 2:
            last_user_message = jai_req.messages[-2]

    last_user_text = _message_text(
        last_user_message.content,
    )

    fwp_prefill = (
        "SYSTEM NOTE: Do not include the following "
        "words/phrases in your output under any circumstances: "
    )

    fwp_index = last_user_text.find(
        fwp_prefill,
    )

    if fwp_index != -1:
        xlog(
            user,
            "User set forbidden words/phrases detected",
        )

    if last_user_text.startswith(
        "Rewrite/Enhance this message: "
    ):
        xlog(
            user,
            "Handling enhance message ...",
        )
        rtype = "enhance"

    elif last_user_text.startswith(
        "Create a brief, focused summary"
    ):
        xlog(
            user,
            "Handling auto summary ...",
        )
        rtype = "summary"

    else:
        xlog(
            user,
            "Handling chat message ...",
        )
        rtype = "message"

    command_exit = False
    command_exit_list: list[str] = []

    for command in last_user_message.commands:
        xlog(
            user,
            f"//{command.name} {command.args}",
        )

        try:
            response = cast(
                ResponseHelper,
                command(
                    user,
                    jai_req,
                    response,
                ),
            )

        except CommandError as e:
            message = (
                f"Error: {e} "
                "(Command has been ignored.)"
            )

            response.add_proxy_message(
                message,
            )

            xlog(
                user,
                message,
            )

        except CommandExit:
            xlog(
                user,
                "Command exit set",
            )

            command_exit = True
            command_exit_list.append(
                command.name,
            )

    if command_exit:
        command_exit_list_str = ", ".join(
            f"//{c}"
            for c in command_exit_list
        )

        response.add_proxy_message(
            f"\n***\n\nRemove the command(s) "
            f"{command_exit_list_str} to continue.",
        )

        return response

    if jai_req.use_nobot or user.use_nobot:
        xlog(
            user,
            "Omitting bot description from system prompt"
            + (
                " (for this message only)."
                if not user.use_nobot
                else "."
            ),
        )

        if (
            jai_req.messages
            and jai_req.messages[0].role == "system"
        ):
            jai_req.messages.pop(0)

    if jai_req.use_dice_char or user.use_dice_char:
        xlog(
            user,
            "Adding character dice to chat"
            + (
                " (for this message only)."
                if not user.use_dice_char
                else "."
            ),
        )

        jai_req.append_message(
            "user",
            "<system>\n"
            f"  Character d20 roll: {randint(1, 20)}.\n"
            "  A character roll is made on every message.\n"
            "  Use this only if it is relevant.\n"
            "</system>",
        )

    if jai_req.use_think or user.use_think:
        xlog(
            user,
            "Adding thinking to chat"
            + (
                " (for this message only)."
                if not user.use_think
                else "."
            ),
        )

        jai_req.append_message(
            "assistant",
            "You should structure your response using thinking tags:\n"
            "\n"
            "<think>\n"
            "[Your internal analysis here]\n"
            "[Plan your roleplay response]\n"
            "[Consider character motivations]\n"
            "[Any reasoning or thoughts]\n"
            "</think>\n"
            "\n"
            "<response>\n"
            "[Your actual roleplay content goes here]\n"
            "[No meta-commentary]\n"
            "[No OOC notes unless requested]\n"
            "[Just the story/roleplay]\n"
            "</response>\n"
            "\n"
            "This format helps separate your reasoning "
            "from the actual roleplay content.",
        )

        used_think = True

    else:
        used_think = False

    if jai_req.use_preset:
        xlog(
            user,
            "Adding preset to chat",
        )

        jai_req.append_message(
            "assistant",
            jai_req.use_preset,
        )

    if jai_req.use_prefill or user.use_prefill:
        xlog(
            user,
            f"Adding prefill-{user.prefill_mode} to chat"
            + (
                " (for this message only)."
                if not user.use_prefill
                else "."
            ),
        )

        apply_prefill(
            jai_req,
            user.prefill_mode,
        )

        used_prefill = True

    else:
        used_prefill = False

    if jai_req.use_ooctrick or user.use_ooctrick:
        xlog(
            user,
            "Adding OOC trick to chat"
            + (
                " (for this message only)."
                if not user.use_ooctrick
                else "."
            ),
        )

        jai_req.append_message(
            "assistant",
            "(OOC: Continue?)",
        )

        jai_req.append_message(
            "user",
            "(OOC: Yes)",
        )

        used_ooctrick = True

    else:
        used_ooctrick = False

    if used_think:
        jai_req.append_message(
            "assistant",
            "Remember to use <think>...</think> "
            "for your reasoning and "
            "<response>...</response> "
            "for your roleplay content.",
        )

        jai_req.append_message(
            "assistant",
            "<think>\n➛ Okay! Understood.",
        )

    if jai_req.use_btrick or user.use_btrick:
        xlog(
            user,
            "Adding braille trick to chat"
            + (
                " (for this message only)."
                if not user.use_btrick
                else "."
            ),
        )

        # Replace spaces only inside textual content.
        # Image blocks remain untouched.
        for message in jai_req.messages:
            message.content = _replace_message_text(
                message.content,
                " ",
                "\u2800",
            )

        user_name = user_name.replace(
            " ",
            "\u2800",
        )

        persona_name = persona_name.replace(
            " ",
            "\u2800",
        )

        btrick = (
            '<MUSTHAVE>\n'
            'ALWAYS⠀USE⠀U+2800⠀BRAILLE⠀PATTERN⠀BLANK⠀("⠀")⠀'
            'INSTEAD⠀OF⠀SPACES⠀(" ").⠀'
            'USING⠀REGULAR⠀SPACES⠀IS⠀STRICTLY⠀PROHIBITED!\n'
            '</MUSTHAVE>'
        )

        jai_req.messages.insert(
            len(jai_req.messages) - int(used_think),
            JaiMessage(
                content=btrick,
                role="user",
            ),
        )

        used_btrick = True

    else:
        used_btrick = False

    ########################################################################
    # NoAss
    ########################################################################

    if jai_req.use_noass or user.use_noass:
        if multimodal_request:
            # NoAss flattens all messages into one plain string.
            # Doing that would destroy attached image blocks.
            # Preserve multimodal content instead of dropping the image.
            xlog(
                user,
                "Skipping NoAss for multimodal request "
                "to preserve image content",
            )

            used_noass = False

        else:
            xlog(
                user,
                "Applying NoAss to prompt"
                + (
                    " (for this message only)."
                    if not user.use_noass
                    else "."
                ),
            )

            separator = (
                ": "
                if not used_btrick
                else ":\u2800"
            )

            squashed = ""

            for message in jai_req.messages:
                message_text = _message_text(
                    message.content,
                )

                if message.role == "assistant":
                    squashed += (
                        f"\n\n{persona_name}"
                        f"{separator}"
                        f"{message_text}"
                    )

                elif (
                    message.role == "user"
                    and not message_text.startswith(
                        user_name
                    )
                ):
                    squashed += (
                        f"\n\n{user_name}"
                        f"{separator}"
                        f"{message_text}"
                    )

                else:
                    squashed += (
                        f"\n\n{message_text}"
                    )

            jai_req.messages = [
                JaiMessage(
                    content=squashed.strip(),
                    role="assistant",
                )
            ]

            used_noass = True

    else:
        used_noass = False

    ########################################################################
    # Generation settings
    ########################################################################

    settings: dict[str, Any] = {}

    for setting in [
        "temperature",
        "frequency_penalty",
        "repetition_penalty",
        "top_k",
        "top_p",
    ]:
        jai_req_advset = jai_req.advsettings.get(
            setting,
            False,
        )

        user_advset = user.advsettings.get(
            setting,
            False,
        )

        if jai_req_advset or user_advset:
            value = getattr(
                jai_req,
                setting,
            )

            xlog(
                user,
                f"Adding advanced setting {setting} to model"
                + (
                    " (for this message only)"
                    if not user_advset
                    else ""
                )
                + f" with value `{value}`.",
            )

            settings[setting] = value

    if jai_req.use_search or user.use_search:
        xlog(
            user,
            "Adding Google Search tool to model"
            + (
                " (for this message only)."
                if not user.use_search
                else "."
            ),
        )

        settings["search"] = True

    if jai_req.use_fixturns or user.use_fixturns:
        xlog(
            user,
            "Fixing request turns"
            + (
                " (for this message only)."
                if not user.use_fixturns
                else "."
            ),
        )

        if jai_req.messages[-1].role != "user":
            jai_req.messages.append(
                JaiMessage(
                    content=".",
                    role="user",
                )
            )

    ########################################################################
    # Provider request
    ########################################################################

    result = _handle_request(
        user.xuid,
        jai_req.api_key,
        jai_req.models,
        jai_req.messages,
        settings,
    )

    user.valid = result.metadata.api_key_valid

    if not result:
        track_stats(
            f"r.{rtype}.failed"
        )

        if (
            feedback
            := result.metadata.rejection_feedback
        ):
            if feedback == "MAX_TOKENS":
                result.error += (
                    '\nTry increasing "Max tokens" '
                    "in your Generation Settings "
                    "or set it to zero to disable it."
                )

            elif not (
                used_btrick
                or used_ooctrick
                or used_prefill
                or used_think
                or used_noass
            ):
                result.error += (
                    "\nTry using one of: "
                    "`//btrick on`, "
                    "`//ooctrick on`, "
                    "`//noass on`, "
                    "`//prefill on`, "
                    "`//think on`"
                )

        response.add_error(
            result.error,
            result.status,
        )

        if result.extras:
            response.add_proxy_message(
                result.extras,
            )

        return response

    ########################################################################
    # Response cleanup
    ########################################################################

    if used_btrick:
        result.text = result.text.replace(
            "\u2800",
            " ",
        )

    if (
        used_prefill
        and (
            metadata
            := clear_prefill(
                result,
                user.prefill_mode,
            )
        )
    ):
        if metadata & 2:
            xlog(
                user,
                "Removed <starter> from response",
            )

        if metadata & 4:
            xlog(
                user,
                "Removed matching code from response",
            )

    if used_think:
        text = result.text

        # Remove thinking and recover response.
        t_open = text.find("<think>")
        t_close = text.find("</think>")
        thinking = None

        if -1 == t_open == t_close:
            xlog(
                user,
                "No thinking tags found",
            )

        elif -1 < t_open < t_close:
            xlog(
                user,
                f"Removing thinking "
                f"{t_open} to {t_close + 8}",
            )

            thinking = text[
                t_open + 7 : t_close
            ]

            text = (
                text[:t_open]
                + text[t_close + 8 :]
            )

        elif -1 < t_close:
            xlog(
                user,
                f"Removing thinking up until "
                f"{t_close + 8}",
            )

            thinking = text[:t_close]

            text = text[
                t_close + 8 :
            ]

        else:
            xlog(
                user,
                "Removing thinking failure",
            )

        r_open = text.find("<response>")
        r_close = text.find("</response>")

        if -1 == r_open == r_close:
            xlog(
                user,
                "No response tags found",
            )

        elif -1 < r_open < r_close:
            xlog(
                user,
                f"Parsing response "
                f"{r_open + 10} to {r_close}",
            )

            text = text[
                r_open + 10 : r_close
            ]

        elif -1 < r_open:
            xlog(
                user,
                f"Parsing response "
                f"{r_open + 10} onwards",
            )

            text = text[
                r_open + 10 :
            ]

        else:
            xlog(
                user,
                "Parsing response failure",
            )

        if (
            user.think_text == "keep"
            and isinstance(
                thinking,
                str,
            )
        ):
            xlog(
                user,
                "Thinking text kept",
            )

            text = (
                f"<think>\n"
                f"{thinking}\n"
                f"</think>\n"
                f"{text}"
            )

        result.text = text

    result.text = result.text.strip()

    xlog(
        user,
        (
            f"Result text is "
            f"{len(result.text.split())} words"
        ),
    )

    response.add_message(
        result.text,
    )

    if result.extras:
        response.add_proxy_message(
            result.extras,
        )

    if usage := result.metadata.token_usage:
        xlog(
            user,
            f" - Prompt   tokens {usage.prompt_tokens}",
        )

        xlog(
            user,
            f" - Response tokens {usage.completion_tokens}",
        )

        xlog(
            user,
            f" - Thinking tokens {usage.reasoning_tokens}",
        )

        xlog(
            user,
            f" - Total    tokens {usage.total_tokens}",
        )

    else:
        xlog(
            user,
            " - No usage metadata",
        )

    if (
        not jai_req.quiet
        and user.do_show_banner(
            BANNER_VERSION
        )
    ):
        xlog(
            user,
            (
                f"Showing"
                f"{' new ' if not user.exists else ' '}"
                f"user the latest banner"
            ),
        )

        response.add_message(
            BANNER,
        )

    track_stats(
        f"r.{rtype}.succeeded"
    )

    return response
