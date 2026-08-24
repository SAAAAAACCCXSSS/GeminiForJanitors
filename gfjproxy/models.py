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
            # OpenAI multimodal format used by Tavo:
            # [
            #   {"type": "text", "text": "..."},
            #   {"type": "image_url", "image_url": {"url": "..."}}
            # ]
            jai_msg.content = content

        elif isinstance(content, str):
            if role == "user":
                jai_msg.commands, jai_msg.content = parse_message(content)
            else:
                jai_msg.content = strip_message(content)

        elif content is not None:
            jai_msg.content = str(content)

        return jai_msg
