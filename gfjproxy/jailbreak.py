"""Assistant prefill + editor framing for explicit presets."""

ENABLED = True

JAILBREAK_PRESETS = {
    "sexstyle",
    "slowhand",
    "pwp",
    "filthy",
}

PREFILLS = {
    "sexstyle": "*Neither moves for a second. Then—*",
    "slowhand": "*Neither moves for a second. Then—*",
    "pwp": "*The room is small. The door is closed. And—*",
    "filthy": "*Neither moves for a second. Then—*",
}

EDITOR_FRAMING = {}

# Используем обычный словарь вместо атрибутов jai_req
_STAGING = {}

def is_enabled(name: str) -> bool:
    if not ENABLED or not name:
        return False
    return name in JAILBREAK_PRESETS

def prepare(name: str, jai_req) -> None:
    if not is_enabled(name):
        return
    prefill = PREFILLS.get(name, "").strip()
    if not prefill:
        return
    _STAGING[id(jai_req)] = (name, prefill)

def finalize(jai_req) -> None:
    data = _STAGING.pop(id(jai_req), None)
    if not data:
        return
    name, prefill = data
    framing = EDITOR_FRAMING.get(name, "").strip()
    if framing:
        for message in reversed(jai_req.messages):
            if message.role == "user":
                message.content = message.content.rstrip() + framing
                break
    jai_req.append_message("assistant", prefill)

def clear(jai_req) -> None:
    _STAGING.pop(id(jai_req), None)

def get_active_preset(jai_req) -> str:
    data = _STAGING.get(id(jai_req))
    return data[0] if data else "none"
