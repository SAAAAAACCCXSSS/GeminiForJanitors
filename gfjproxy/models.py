"""Data models.

The name of some fields in here may not match with the source they are from."""

import re
from dataclasses import dataclass, field
from json import loads

from .commands import Command, parse_message, strip_message
from .utils import comma_split

################################################################################


_COMMAND_SPACE_TRANSLATION = str.maketrans(
    {
        "\u2800": " ",  # Braille Pattern Blank
        "\u00a0": " ",  # NO-BREAK SPACE
        "\u2007": " ",  # FIGURE SPACE
        "\u202f": " ",  # NARROW NO-BREAK SPACE
        "\u200b": "",   # ZERO WIDTH SPACE
        "\ufeff": "",   # BOM / ZERO WIDTH NO-BREAK SPACE
        "\u2060": "",   # WORD JOINER
    }
)

# Some frontends can serialize a user turn as:
#   Anna: //aboutme
# instead of sending only:
#   //aboutme
#
# Only treat a speaker prefix specially when what follows it is a command.
_SPEAKER_COMMAND_PREFIX = re.compile(
    r"^[^:\n]{1,80}:\s*(?=(?://|!!))"
)


def _parse_user_text(text: str) -> tuple[list[Command], str]:
    """Parse GFJ commands safely from a user-role text message.

    Supported forms:
        //aboutme
        //btrick on
        !!aboutme
        !!btrick on
        Anna: //aboutme
        Anna: !!btrick on

    Commands are recognized only at the beginning of a line (optionally after
    a simple "speaker:" prefix). This prevents long Tavo prompt text that merely
    mentions //btrick or //fixturns from being executed accidentally.
    """

    if not isinstance(text, str):
        return [], str(text)

    original = text.strip()
    normalized = original.translate(_COMMAND_SPACE_TRANSLATION)

    commands: list[Command] = []
    kept_lines: list[str] = []

    for raw_line in normalized.splitlines() or [normalized]:
        candidate = raw_line.strip()

        prefix_match = _SPEAKER_COMMAND_PREFIX.match(candidate)
        if prefix_match:
            candidate = candidate[prefix_match.end():].lstrip()

        if candidate.startswith("!!"):
            candidate = "//" + candidate[2:]

        if candidate.startswith("//"):
            parsed_commands, clean_text = parse_message(candidate)

            if parsed_commands:
                commands.extend(parsed_commands)

                # Preserve any ordinary residual text that followed commands.
                if clean_text:
                    kept_lines.append(clean_text)

                continue

        kept_lines.append(raw_line)

    if commands:
        return commands, "\n".join(kept_lines).strip()

    return [], original


def _parse_structured_content(
    role: str,
    content: list,
) -> tuple[list[Command], list]:
    """Parse OpenAI/Tavo structured content while preserving non-text blocks."""

    commands: list[Command] = []
    parsed_content: list = []

    for block in content:
        if isinstance(block, str):
            if role == "user":
                block_commands, clean_text = _parse_user_text(block)
                commands.extend(block_commands)
                parsed_content.append(clean_text)
            else:
                parsed_content.append(strip_message(block))
            continue

        if not isinstance(block, dict):
            parsed_content.append(block)
            continue

        new_block = dict(block)
        text = new_block.get("text")

        if isinstance(text, str):
            if role == "user":
                block_commands, clean_text = _parse_user_text(text)
                commands.extend(block_commands)
                new_block["text"] = clean_text
            else:
                new_block["text"] = strip_message(text)

        # image_url / input_image / inline data and every other non-text field
        # are preserved untouched.
        parsed_content.append(new_block)

    return commands, parsed_content


################################################################################


@dataclass(kw_only=True, slots=True)
class JaiMessage:
    """JanitorAI / OpenAI-compatible Message."""

    commands: list[Command] = field(default_factory=list)
    content: str | list = "."
    role: str = "user"

    @staticmethod
    def parse(data: dict | str):
        if isinstance(data, str):
            data = loads(data)

        if not isinstance(data, dict):
            raise TypeError("Invalid data")

        jai_msg = JaiMessage()

        role = data.get("role")
        if role:
            jai_msg.role = role

        content = data.get("content")

        if isinstance(content, list):
            jai_msg.commands, jai_msg.content = _parse_structured_content(
                jai_msg.role,
                content,
            )

        elif isinstance(content, str):
            if jai_msg.role == "user":
                jai_msg.commands, jai_msg.content = _parse_user_text(content)
            else:
                jai_msg.content = strip_message(content)

        elif content is not None:
            jai_msg.content = str(content)

        return jai_msg


@dataclass(kw_only=True, slots=True)
class JaiRequest:
    """JanitorAI Request."""

    # Authorization header
    api_key: str = ""
    api_key_index: int = 1
    api_key_count: int = 1

    # Request body
    max_tokens: int = 0
    messages: list[JaiMessage] = field(default_factory=list)
    models: dict[str, str] = field(default_factory=dict)
    stream: bool = False
    temperature: float = 0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 0.0
    top_k: int = 0
    top_p: float = 0.0

    # Request URL
    quiet: bool = False
    quiet_commands: bool = False  # Only for testing

    # Commands
    use_fixturns: bool = False
    use_btrick: bool = False
    use_dice_char: bool = False
    use_noass: bool = False
    use_nobot: bool = False
    use_ooctrick: bool = False
    use_prefill: bool = False
    use_preset: str | None = None
    use_search: bool = False
    use_think: bool = False

    advsettings: dict[str, bool] = field(default_factory=dict)

    def append_message(self, role: str, content: str):
        self.messages.append(
            JaiMessage(
                content=content,
                role=role,
            )
        )

    @staticmethod
    def parse(data: dict | str):
        if isinstance(data, str):
            data = loads(data)

        if not isinstance(data, dict):
            raise TypeError("Invalid data")

        jai_req = JaiRequest()

        if max_tokens := data.get("max_tokens"):
            jai_req.max_tokens = max_tokens

        if messages := data.get("messages"):
            jai_req.messages = [
                JaiMessage.parse(jai_msg)
                for jai_msg in messages
            ]

        if models := data.get("model"):
            for model in comma_split(models.lower()):
                if "/" in model:
                    provider, model_name = model.split(
                        "/",
                        maxsplit=1,
                    )
                    jai_req.models[provider] = model_name

                elif model.startswith(("gemini-", "gemma-")):
                    jai_req.models["google"] = model

                elif model.startswith("deepseek-"):
                    jai_req.models["deepseek"] = model

                elif model.startswith("mistral-"):
                    jai_req.models["mistral"] = model

                else:
                    if unknown := jai_req.models.get("unknown"):
                        jai_req.models["unknown"] = (
                            f"{unknown}, {model}"
                        )
                    else:
                        jai_req.models["unknown"] = model

        if stream := data.get("stream"):
            jai_req.stream = stream

        if temperature := data.get("temperature"):
            jai_req.temperature = temperature

        if top_k := data.get("top_k"):
            jai_req.top_k = top_k

        if top_p := data.get("top_p"):
            jai_req.top_p = top_p

        if frequency_penalty := data.get("frequency_penalty"):
            jai_req.frequency_penalty = frequency_penalty

        if repetition_penalty := data.get("repetition_penalty"):
            jai_req.repetition_penalty = repetition_penalty

        return jai_req


################################################################################


@dataclass(kw_only=True, slots=True)
class JaiResultTokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(kw_only=True, slots=True)
class JaiResultMetadata:
    "JanitorAI Generate Content Result Metadata"

    api_key_valid: bool = True
    rejection_feedback: str = ""
    token_usage: JaiResultTokenUsage | None = None


@dataclass(init=False, kw_only=True, slots=True)
class JaiResult:
    "JanitorAI Generate Content Result"

    status: int
    text: str
    error: str
    extras: str
    metadata: JaiResultMetadata

    def __init__(
        self,
        status: int,
        message: str,
        *,
        extras: str = "",
        metadata: JaiResultMetadata | None = None,
    ):
        self.status = status

        if status == 200:
            self.text = message
            self.error = ""
        else:
            self.text = ""
            self.error = message

        self.extras = extras
        self.metadata = metadata or JaiResultMetadata()

    def __bool__(self) -> bool:
        return self.status == 200


################################################################################
