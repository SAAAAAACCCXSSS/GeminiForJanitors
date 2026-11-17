"""Parse, persist and apply narrative controls without Chat Memory."""

import re
from copy import deepcopy
from dataclasses import dataclass

from .rp_rules import (
    ALIASES,
    COMMAND_NAMES,
    CORE_RULES,
    LOVE,
    LOVE_EXPRESSIONS,
    MODES,
    THEMES,
    TOGGLES,
)

MAX_VALUE_LENGTH = 2000
MAX_ENTRIES = 96
MAX_STATE_LENGTH = 24000
MAX_CAST = 16
ON = {"on", "active", "this"}
OFF = {"off", "nonactive"}
_NAMES = re.compile(
    "(?:"
    + "|".join(
        re.escape(name).replace(r"\ ", r"[ \t]+")
        for name in sorted(COMMAND_NAMES, key=len, reverse=True)
    )
    + r")(?![\w+])",
    re.IGNORECASE,
)
_WORD = re.compile(r"[\w+-]+")


class RoleplayError(ValueError):
    """An invalid narrative control; its state must not be saved."""


@dataclass
class Directive:
    name: str
    args: str = ""
    value: str = ""
    payload: str = ""
    target: str = ""
    temporary: bool = False
    error: str = ""


def _space(text, pos):
    while pos < len(text) and text[pos] in " \t":
        pos += 1
    return pos


def _word(text, pos):
    start = _space(text, pos)
    match = _WORD.match(text, start)
    return (match.group().lower(), match.end()) if match else ("", pos)


def _group(text, pos):
    opening = text[pos]
    closing = ")" if opening == "(" else "]"
    depth, quoted, escaped = 1, False, False
    for index in range(pos + 1, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif not quoted:
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if not depth:
                    return text[pos + 1 : index].strip(), index + 1, ""
    # Recover at the line boundary so a typo cannot swallow later commands.
    end = text.find("\n", pos)
    end = len(text) if end == -1 else end
    return "", end, f"Unclosed {opening}{closing} in command."


def parse_directive(text: str, pos: int) -> tuple[Directive, int] | None:
    """Parse one RP command after its // marker, preserving following prose."""
    match = _NAMES.match(text, pos)
    if not match:
        return None
    name = " ".join(match.group().lower().split())
    directive = Directive(name)
    args_start = end = match.end()
    value, after_value = _word(text, end)

    if name == "recap":
        if value in {"short", "full"}:
            directive.value, end = value, after_value
    elif name == "writer mode":
        if value in ON | OFF:
            directive.value, end = value, after_value
        else:
            directive.value = "active"
    elif name in {"interlude", "retcon", "boundaries"}:
        if value in OFF:
            directive.value, end = value, after_value
    else:
        directive.value, end = value, after_value
        if name == "format" and value == "social":
            second, after_second = _word(text, end)
            if second == "media":
                directive.value, end = "social media", after_second

    # Themes allow //angst theme and //angst theme off/this.
    if directive.value == "theme":
        action, after_action = _word(text, end)
        if action in ON | OFF:
            directive.value = f"theme {action}"
            end = after_action

    seen = set()
    for _ in range(4):
        start = _space(text, end)
        if start < len(text) and text[start] in "([":
            kind = text[start]
            if kind in seen:
                break
            seen.add(kind)
            content, end, error = _group(text, start)
            if kind == "(":
                directive.payload = content
            else:
                directive.target = content
                if not content and not error:
                    error = "A character target in brackets must not be empty."
            directive.error = error
            if error:
                break
        else:
            word, next_end = _word(text, end)
            if word == "this" and "this" not in seen:
                directive.temporary = True
                seen.add("this")
                end = next_end
            else:
                break
    directive.temporary |= directive.value in {"this", "theme this"}
    directive.args = text[args_start:end].strip()
    return directive, end


def _state(user, jai_req):
    if jai_req.rp_state is None:
        jai_req.rp_state = deepcopy(user.rp_settings)
    return jai_req.rp_state


def _validate_state(state):
    if len(state) > MAX_ENTRIES or len(str(state)) > MAX_STATE_LENGTH:
        raise RoleplayError(
            "Too many RP settings. Remove unused entries or use //rp reset."
        )
    if len(state.get("cocharacters", [])) > MAX_CAST:
        raise RoleplayError(f"At most {MAX_CAST} added main characters are supported.")


def _change(
    user, jai_req, changes, temporary=False, *, enable=True, saved_changes=None
):
    """Validate both snapshots before publishing either (including removals)."""
    effective = deepcopy(_state(user, jai_req))
    saved = deepcopy(user.rp_settings)
    for state, updates in [(effective, changes)] + (
        []
        if temporary
        else [(saved, changes if saved_changes is None else saved_changes)]
    ):
        for key, value in updates.items():
            if value is None:
                state.pop(key, None)
            else:
                state[key] = deepcopy(value)
    _validate_state(effective)
    _validate_state(saved)
    jai_req.rp_state = effective
    if not temporary:
        user.rp_settings = saved
    if enable:
        jai_req.rp_enabled = True
        jai_req.use_rp = True
        if not temporary:
            user.use_rp = True


def _key(name, target):
    return f"{name}::{target.casefold()}" if target else name


def _entry(directive, value=None):
    return {
        "value": directive.value if value is None else value,
        "payload": directive.payload,
        "target": directive.target,
    }


def _names(payload):
    names = [name.strip() for name in payload.split(",") if name.strip()]
    if not names or any(len(name) > 200 for name in names):
        raise RoleplayError(
            "Specify character names in parentheses (up to 200 characters each)."
        )
    return names


def _message_text(jai_req):
    return next(
        (
            message.content.strip()
            for message in reversed(jai_req.messages)
            if message.role == "user"
        ),
        "",
    )


def status_text(user, jai_req) -> str:
    effective = _state(user, jai_req)
    saved = user.rp_settings
    enabled = jai_req.rp_enabled
    if enabled is None:
        enabled = jai_req.use_rp or user.use_rp
    lines = [
        f"RP: {'on' if enabled else 'off'}; saved RP: {'on' if user.use_rp else 'off'}.",
        "Saved settings use the primary API key across chats; no Chat Memory is required.",
    ]
    for key, value in sorted(effective.items()):
        temporary = " (this request)" if saved.get(key) != value else ""
        if key == "cocharacters":
            lines.append("Main cast: " + ", ".join(value) + temporary)
        elif isinstance(value, dict):
            label = key.split("::", 1)[0]
            target = f" [{value['target']}]" if value.get("target") else ""
            payload = f" ({value['payload']})" if value.get("payload") else ""
            lines.append(f"- {label}: {value['value']}{payload}{target}{temporary}")
    if not effective:
        lines.append("No additional RP settings or cast.")
    lines.append(
        "//rp off pauses; //rp on resumes; //rp reset clears settings and added cast."
    )
    return "\n".join(lines)


def execute(d: Directive, user, jai_req, response):
    if d.error:
        raise RoleplayError(d.error)
    if any(len(value) > MAX_VALUE_LENGTH for value in (d.payload, d.target)):
        raise RoleplayError(
            f"RP descriptions must be at most {MAX_VALUE_LENGTH} characters."
        )
    if d.target and len(d.target) > 200:
        raise RoleplayError("A target name must be at most 200 characters.")

    name, value = d.name, d.value
    state = _state(user, jai_req)
    key = _key(name, d.target)

    if name == "rp":
        if d.payload or d.target or (d.temporary and value != "this"):
            raise RoleplayError(
                "Use //rp on|off|this|status|reset without a target or description."
            )
        if value == "status":
            response.add_proxy_message(status_text(user, jai_req))
            jai_req.rp_status_only = True
            return response
        if value == "reset":
            user.rp_settings = {}
            jai_req.rp_state = {}
            user.use_rp = jai_req.use_rp = False
            jai_req.rp_enabled = False
            jai_req.rp_tasks.clear()
            jai_req.rp_generate = False
            confirmation = (
                "RP settings and added cast reset. Chat history is unchanged."
            )
        elif value in {"on", "off", "this"}:
            enabled = value != "off"
            jai_req.rp_enabled = jai_req.use_rp = enabled
            if value != "this":
                user.use_rp = enabled
            confirmation = f"Canonical RP mode {'enabled' if enabled else 'disabled'}"
            confirmation += (
                " (for this message only)."
                if value == "this"
                else ". Saved for the primary API key."
            )
        else:
            raise RoleplayError("//rp only accepts on|off|this|status|reset.")
    elif value.startswith("theme") and name in THEMES:
        action = value.removeprefix("theme").strip() or "on"
        if action not in ON | OFF:
            raise RoleplayError("Use //NAME theme [off|this].")
        key = _key("theme " + name, d.target)
        _change(
            user,
            jai_req,
            {key: None if action in OFF else _entry(d, name)},
            d.temporary,
            enable=action not in OFF,
        )
        confirmation = f"Theme {name} {'disabled' if action in OFF else 'enabled'}."
    elif name in ALIASES and name in {"chat", "call", "social media"}:
        if value not in ON | OFF:
            raise RoleplayError(f"//{name} accepts active|nonactive|on|off|this.")
        if d.target or d.payload:
            raise RoleplayError(
                "Output format controls do not accept a target or description."
            )
        current = state.get("format", {}).get("value")
        changes = (
            {"format": _entry(d, name)}
            if value in ON
            else ({"format": None} if current == name else {})
        )
        saved_format = user.rp_settings.get("format", {}).get("value")
        saved_changes = (
            changes
            if value in ON
            else ({"format": None} if saved_format == name else {})
        )
        _change(
            user,
            jai_req,
            changes,
            d.temporary,
            enable=value in ON,
            saved_changes=saved_changes,
        )
        confirmation = f"Format {name} {'enabled' if value in ON else 'disabled'}."
    elif (
        name in TOGGLES
        or name in {"humanize", "writer mode"}
        or (name == "prose" and value in ON | OFF)
    ):
        canonical = ALIASES.get(name, name)
        canonical = "quality" if name == "prose" else canonical
        if value not in ON | OFF:
            raise RoleplayError(f"//{name} accepts active|nonactive|on|off|this.")
        if d.payload:
            raise RoleplayError(
                f"//{name} does not take a description; use [Name] to select a character."
            )
        _change(
            user,
            jai_req,
            {_key(canonical, d.target): None if value in OFF else _entry(d, "on")},
            d.temporary,
            enable=value in ON,
        )
        confirmation = f"{canonical} {'enabled' if value in ON else 'disabled'}."
        if canonical == "scenario select" and value in ON:
            jai_req.rp_generate = True
    elif name in MODES:
        if value not in MODES[name].options and value not in OFF:
            raise RoleplayError(
                f"//{name} only accepts {'|'.join(MODES[name].options)}|off."
            )
        if d.payload or (name == "format" and d.target):
            raise RoleplayError(f"//{name} does not accept that description or target.")
        remove = value in OFF and value not in MODES[name].options
        _change(
            user,
            jai_req,
            {key: None if remove else _entry(d)},
            d.temporary,
            enable=not remove,
        )
        confirmation = f"{name}: {value}."
    elif name == "love":
        if value not in set(LOVE) | OFF:
            raise RoleplayError("//love accepts hidden|active|semi-active|off.")
        expressions = [
            item.strip().lower() for item in d.payload.split(",") if item.strip()
        ]
        if any(item not in LOVE_EXPRESSIONS for item in expressions):
            raise RoleplayError("Unknown love expression; see //help rp.")
        if value == "semi-active" and not expressions:
            raise RoleplayError(
                "Specify expressions, e.g. //love semi-active (care, protectiveness)."
            )
        if d.payload and value != "semi-active":
            raise RoleplayError(
                "Parenthesized expressions are supported with //love semi-active."
            )
        _change(
            user,
            jai_req,
            {key: None if value in OFF else _entry(d)},
            d.temporary,
            enable=value not in OFF,
        )
        confirmation = f"love: {value}."
    elif name in {"npc", "npcs", "cochar", "cochars"}:
        if d.target or value not in ON | OFF:
            raise RoleplayError(
                f"Use //{name} active|nonactive|this (names or description)."
            )
        if name in {"npc", "npcs"}:
            changes = {"supporting cast": None if value in OFF else _entry(d, name)}
        else:
            names = _names(d.payload) if d.payload else []
            if value in ON:
                if not names:
                    raise RoleplayError(
                        "Specify the added main character(s) in parentheses."
                    )
                if name == "cochar" and len(names) != 1:
                    raise RoleplayError("Use //cochars for several names.")

            def update_cast(snapshot):
                cast = list(snapshot.get("cocharacters", []))
                if value in ON:
                    for character in names:
                        if character.casefold() not in {
                            item.casefold() for item in cast
                        }:
                            cast.append(character)
                elif names:
                    cast = [
                        item
                        for item in cast
                        if item.casefold() not in {n.casefold() for n in names}
                    ]
                else:
                    cast = []
                return {"cocharacters": cast or None}

            changes = update_cast(state)
        saved_changes = (
            update_cast(user.rp_settings) if name in {"cochar", "cochars"} else changes
        )
        _change(
            user,
            jai_req,
            changes,
            d.temporary,
            enable=value in ON,
            saved_changes=saved_changes,
        )
        confirmation = (
            f"{name} {'enabled' if value in ON else 'disabled'}"
            + (f" ({d.payload})" if d.payload else "")
            + "."
        )
    elif name == "cast":
        if (
            d.target
            or (value != "focus" and value not in OFF)
            or (value == "focus" and not d.payload)
        ):
            raise RoleplayError("Use //cast focus (Name, Name) or //cast off.")
        _change(
            user,
            jai_req,
            {"cast focus": None if value in OFF else _entry(d)},
            d.temporary,
            enable=value not in OFF,
        )
        confirmation = "Cast focus updated."
    elif name in {
        "plot",
        "relationship",
        "character",
        "memory",
        "retcon",
        "boundaries",
    }:
        if (
            name in {"plot", "relationship", "character", "memory"}
            and value not in {"override"} | OFF
        ):
            raise RoleplayError(f"Use //{name} override (change) or //{name} off.")
        payload = d.payload or _message_text(jai_req)
        if value not in OFF and not payload:
            raise RoleplayError(
                f"Specify the fictional change or constraints for //{name} in parentheses or in your message."
            )
        if len(payload) > MAX_VALUE_LENGTH:
            raise RoleplayError(
                f"Author directions must be at most {MAX_VALUE_LENGTH} characters."
            )
        d.payload = payload if value not in OFF else ""
        if name in {"memory", "retcon"}:
            # Keep successive facts, with bounded storage and case-stable targets.
            prefix = _key(name, d.target) + " #"
            changes = (
                {
                    k: None
                    for k in state.keys() | user.rp_settings.keys()
                    if k.startswith(prefix)
                }
                if value in OFF
                else {}
            )
            if value not in OFF:
                existing = [entry for k, entry in state.items() if k.startswith(prefix)]
                if not any(entry.get("payload") == payload for entry in existing):
                    index = (
                        max(
                            (
                                int(k.rsplit(" #", 1)[1])
                                for k in state
                                if k.startswith(prefix)
                            ),
                            default=0,
                        )
                        + 1
                    )
                    changes[prefix + str(index)] = _entry(d, "override")
        else:
            changes = {key: None if value in OFF else _entry(d, "override")}
        _change(user, jai_req, changes, d.temporary, enable=value not in OFF)
        confirmation = (
            f"{name}: {'cleared' if value in OFF else 'author direction saved'}."
        )
    elif name == "scene":
        if d.target or value not in {"stay", "next"} | OFF:
            raise RoleplayError(
                "Use //scene stay, //scene next (direction), or //scene off."
            )
        if value == "next":
            direction = d.payload or _message_text(jai_req)
            if not direction:
                raise RoleplayError("Specify where the next scene should begin.")
            _change(user, jai_req, {"scene": None}, d.temporary)
            jai_req.rp_tasks.append(
                "Transition once to the requested scene: "
                + direction
                + ". Do not invent important choices or dialogue for the user during the gap."
            )
        else:
            _change(
                user,
                jai_req,
                {"scene": None if value in OFF else _entry(d)},
                d.temporary,
                enable=value not in OFF,
            )
        confirmation = "Scene direction updated."
    elif name in {"recap", "interlude"}:
        if d.target or (name == "interlude" and not d.payload):
            raise RoleplayError(
                "Use //recap [short|full] or //interlude (character; episode)."
            )
        if name == "recap":
            task = "For this response only, summarize established events, chronology, places, relationships, promises and unresolved conflicts. Do not advance the story, invent facts or claim to edit Chat Memory. Use clear headings and distinguish uncertainty."
            task += (
                " Keep it compact." if value != "full" else " Include relevant detail."
            )
        else:
            task = (
                "For this response only, write an off-screen fictional interlude: "
                + d.payload
                + ". Do not control the user or make private information available to absent characters."
            )
        jai_req.rp_tasks.append(task)
        confirmation = f"{name}: requested for this response."
    else:
        raise RoleplayError(f"Use //help rp for //{name} syntax.")

    if d.temporary:
        jai_req.rp_generate = True
    if not jai_req.quiet_commands:
        target = f" [{d.target}]" if d.target else ""
        scope = " (this request)" if d.temporary and name != "rp" else ""
        response.add_proxy_message(confirmation + target + scope)
    return response


def enabled(user, jai_req):
    if jai_req.rp_enabled is not None:
        return jai_req.rp_enabled
    return jai_req.use_rp or user.use_rp


def build_prompt(user, jai_req) -> str:
    """Build one compact block, never a transcript of previous commands."""
    lines = []
    if enabled(user, jai_req):
        for key, entry in _state(user, jai_req).items():
            name = key.split("::", 1)[0].split(" #", 1)[0]
            if name == "cocharacters":
                lines.append(
                    "Persistent assistant-controlled MAIN CAST alongside the original character: "
                    + "; ".join(entry)
                    + ". Give each independent agency and narrative importance over time."
                )
                continue
            value, payload, target = (
                entry["value"],
                entry.get("payload", ""),
                entry.get("target", ""),
            )
            if name in MODES:
                rule = MODES[name].options[value]
            elif name in TOGGLES:
                rule = TOGGLES[name]
            elif name == "love":
                rule = LOVE[value]
                if payload:
                    rule += (
                        " Expressions: "
                        + payload
                        + ". Keep expressions coherent with the character and established relationship."
                    )
            elif name.startswith("theme "):
                rule = "Story theme: " + name[6:] + ". " + THEMES[value]
            elif name == "supporting cast":
                rule = (
                    "Actively portray "
                    + (
                        "one contextually relevant supporting character"
                        if value == "npc"
                        else "multiple contextually relevant supporting characters"
                    )
                    + ". Infer reasonable missing details; preserve the original character's involvement."
                )
                if payload:
                    rule += " Specification: " + payload
            elif name == "cast focus":
                rule = (
                    "Focus this ongoing episode on: "
                    + payload
                    + ". Other main cast members retain their status."
                )
            elif name == "scene":
                rule = "Stay within the current scene; do not skip to a later time or new location without an explicit transition request."
            else:
                rule = f"Explicit fictional {name} direction (retain until changed): {payload}"
            if target:
                rule = f"Character [{target}]: {rule} Emotional directions concern their relationship to the user unless explicitly specified otherwise."
            lines.append(rule)
    if jai_req.rp_tasks:
        lines.extend(jai_req.rp_tasks)
    return (
        "<roleplay_controls>\n"
        + CORE_RULES
        + "\n\n"
        + "\n".join("- " + line for line in lines)
        + "\n</roleplay_controls>"
        if lines
        else ""
    )
