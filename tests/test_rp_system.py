"""Behavioral coverage for parsing, persistence, scopes and narrative dispatch."""

from copy import deepcopy

import pytest

from gfjproxy.commands import CommandError, parse_message
from gfjproxy.handlers import handle_chat_message
from gfjproxy.models import JaiMessage, JaiRequest, JaiResult
from gfjproxy.roleplay import build_prompt
from gfjproxy.rp_rules import MODES, THEMES, TOGGLES
from gfjproxy.utils import ResponseHelper
from gfjproxy.xuiduser import XUID, LocalUserStorage, UserSettings


@pytest.fixture
def run(mocker):
    storage = LocalUserStorage()
    provider = mocker.patch(
        "gfjproxy.handlers._handle_request",
        return_value=JaiResult(200, "Story response."),
    )

    def send(
        text,
        *,
        key="primary",
        system="Original character description.",
        history=(),
        prefill=False,
        quiet=False,
    ):
        user = UserSettings(storage, XUID(key, "test-salt"))
        messages = [JaiMessage(role="system", content=system)] if system else []
        messages += list(history) + [
            JaiMessage.parse({"role": "user", "content": text})
        ]
        if prefill:
            messages.append(JaiMessage(role="assistant", content="Prefill."))
        request = JaiRequest(messages=messages, quiet_commands=quiet)
        provider.reset_mock()
        response = handle_chat_message(user, request, ResponseHelper())
        user.save()
        return user, request, response, provider

    return send


@pytest.mark.parametrize(
    "text, expected, remaining",
    [
        (
            "Before //LOVE semi-active (care, protectiveness) [Pantalone] after.",
            [("love", "semi-active (care, protectiveness) [Pantalone]")],
            "Before after.",
        ),
        (
            "//SOCIAL MEDIA active //prefill on Continue.",
            [("social media", "active"), ("prefill", "on")],
            "Continue.",
        ),
        (
            "//pace slow//detail rich Continue.",
            [("pace", "slow"), ("detail", "rich")],
            "Continue.",
        ),
        (
            "//cochars active (Pantalone, Dr. Ratio)\nHello.",
            [("cochars", "active (Pantalone, Dr. Ratio)")],
            "Hello.",
        ),
        (
            '//memory override (The letter says "wait (until dawn)".) Continue.',
            [("memory", 'override (The letter says "wait (until dawn)".)')],
            "Continue.",
        ),
        (
            "//prose classical this [Dainsleif] Hello",
            [("prose", "classical this [Dainsleif]")],
            "Hello",
        ),
        (
            "https://example.com//pace slow //unknown active",
            [],
            "https://example.com//pace slow //unknown active",
        ),
        ("//recap Please summarize.", [("recap", "")], "Please summarize."),
    ],
)
def test_parser_preserves_prose_and_command_order(text, expected, remaining):
    commands, content = parse_message(text)
    assert [(command.name, command.args) for command in commands] == expected
    assert content == remaining


@pytest.mark.parametrize(
    "name, value",
    [(name, value) for name, mode in MODES.items() for value in mode.options],
)
def test_mode_controls_reach_model_and_persist(run, name, value):
    _, request, response, provider = run(f"//{name} {value} Continue.")
    assert response.status == 200
    provider.assert_called_once()
    assert MODES[name].options[value] in build_prompt(*run("Continue.")[:2])
    assert request.messages[0].content.startswith("Original character description.")
    assert f"//{name}" not in request.messages[-1].content


@pytest.mark.parametrize("name", list(THEMES))
def test_every_theme_can_be_enabled_and_removed(run, name):
    user, request, _, provider = run(f"//{name} theme")
    provider.assert_not_called()
    assert f"Story theme: {name}." in build_prompt(user, request)
    assert f"Story theme: {name}." in build_prompt(*run("Continue.")[:2])
    run(f"//{name} theme off")
    assert f"Story theme: {name}." not in build_prompt(*run("Continue.")[:2])


@pytest.mark.parametrize("name", list(TOGGLES))
def test_toggles_enable_temporarily_without_saving(run, name):
    user, request, _, _ = run(f"//{name} this Continue.")
    assert TOGGLES[name] in build_prompt(user, request)
    assert user.rp_settings == {}
    assert not user.use_rp
    assert TOGGLES[name] not in build_prompt(*run("Continue.")[:2])


def test_saved_settings_survive_one_shot_overrides_and_pause(run):
    run("//pace slow //detail rich")
    user, request, _, _ = run("//pace fast this //initiative high Continue.")
    assert user.rp_settings["pace"]["value"] == "slow"
    assert request.rp_state["pace"]["value"] == "fast"
    run("//rp off")
    assert build_prompt(*run("Continue.")[:2]) == ""
    run("//rp on")
    assert "small developments" in build_prompt(*run("Continue.")[:2])
    assert "specific sensory" in build_prompt(*run("Continue.")[:2])


def test_persistence_scope_uses_primary_key(run):
    run("//cochar active (Pantalone) //pace slow", key="one")
    assert "Pantalone" in build_prompt(
        *run("A new chat.", key="one", system="Different bot.")[:2]
    )
    assert "Pantalone" not in build_prompt(*run("Continue.", key="two")[:2])


def test_cast_accumulates_deduplicates_and_removes_without_leaking_temporary_members(
    run,
):
    run("//cochar active (Pantalone)")
    user, request, _, _ = run(
        "//cochar this (Dottore) //cochars active (pantalone, Capitano) Continue."
    )
    assert user.rp_settings["cocharacters"] == ["Pantalone", "Capitano"]
    assert request.rp_state["cocharacters"] == ["Pantalone", "Dottore", "Capitano"]
    run("//cochar nonactive (PANTALONE)")
    assert run("Continue.")[0].rp_settings["cocharacters"] == ["Capitano"]
    run("//cochars off")
    assert "cocharacters" not in run("Continue.")[0].rp_settings


def test_targeted_emotions_do_not_collapse_into_one_setting(run):
    user, request, _, _ = run(
        "//love hidden [Pantalone] //hatred active [Dottore] //love semi-active (care, protectiveness) [Capitano] Continue."
    )
    prompt = build_prompt(user, request)
    assert "Character [Pantalone]: Portray restrained" in prompt
    assert "Character [Dottore]: Portray motivated hostility" in prompt
    assert "Character [Capitano]: Let affection" in prompt
    run("//love off [pantalone]")
    prompt = build_prompt(*run("Continue.")[:2])
    assert "Character [Pantalone]" not in prompt
    assert "Character [Capitano]" in prompt


def test_formats_replace_each_other_and_clear_the_correct_saved_mode(run):
    run("//call active")
    user, request, _, _ = run("//chat this //call nonactive Continue.")
    assert request.rp_state["format"]["value"] == "chat"
    assert "format" not in user.rp_settings
    run("//format letters")
    run("//social media active")
    assert run("Continue.")[0].rp_settings["format"]["value"] == "social media"


def test_author_facts_accumulate_and_recover_from_temporary_clear(run):
    run("//memory override (The letter is sealed.)")
    run("//memory override (Nobody knows the sender.)")
    prompt = build_prompt(*run("Continue.")[:2])
    assert "The letter is sealed." in prompt
    assert "Nobody knows the sender." in prompt
    run("//memory off this //memory off")
    assert "letter is sealed" not in build_prompt(*run("Continue.")[:2])


def test_author_direction_uses_user_message_with_prefill(run):
    user, request, _, _ = run(
        "//plot override Move the scene to the winter palace.", prefill=True
    )
    assert "Move the scene to the winter palace." in build_prompt(user, request)
    assert "Prefill." not in build_prompt(user, request)


@pytest.mark.parametrize(
    "command, needle",
    [
        ("//recap", "summarize established events"),
        ("//recap full", "Include relevant detail"),
        (
            "//interlude (Pantalone; a private conversation)",
            "off-screen fictional interlude",
        ),
        ("//scene next (the next morning)", "Transition once"),
    ],
)
def test_one_shot_tasks_generate_without_becoming_saved_modes(run, command, needle):
    user, request, _, provider = run(command)
    provider.assert_called_once()
    assert needle in request.messages[0].content
    assert user.rp_settings == {}
    assert needle not in build_prompt(*run("Continue.")[:2])


def test_status_reset_and_settings_only_messages_do_not_call_model(run):
    run("//cochar active (Pantalone) //pace slow")
    _, _, response, provider = run("//rp status")
    provider.assert_not_called()
    assert "Pantalone" in response.message and "pace: slow" in response.message
    user, _, response, provider = run("//rp reset")
    provider.assert_not_called()
    assert user.rp_settings == {} and not user.use_rp
    assert "Chat history is unchanged" in response.message


@pytest.mark.parametrize(
    "text",
    [
        "//pace warp",
        "//love semi-active (unknown)",
        "//cochar active",
        "//cochar active (one, two)",
        "//cochar active (unclosed\n//rp status",
        "//love hidden []",
        "//cast focus",
        "//interlude",
        "//scene next",
        "//boundaries",
        "//voice canon (bad payload)",
        "//chat active [Pantalone]",
        "//memory override (" + "x" * 2001 + ")",
    ],
)
def test_invalid_commands_are_visible_and_do_not_change_saved_state(run, text):
    original = deepcopy(run("//pace slow")[0].rp_settings)
    user, _, response, _ = run(text)
    assert "Error:" in response.message
    assert user.rp_settings == original


def test_old_commands_and_history_are_not_reexecuted(run):
    old = JaiMessage.parse({"role": "user", "content": "//pace fast Earlier turn."})
    user, _, _, _ = run("//pace slow Continue.", history=[old])
    assert user.rp_settings["pace"]["value"] == "slow"
    _, request, _, provider = run("//prefill this Continue.")
    provider.assert_called_once()
    assert request.use_prefill


def test_help_and_quality_aliases(run):
    for alias in ("//humanize active", "//prose active", "//writer mode"):
        user, request, _, _ = run(alias)
        assert "human-sounding fiction" in build_prompt(user, request)
    _, _, response, provider = run("//help rp")
    provider.assert_not_called()
    assert "COMMAND SYSTEM by dainsleifswife" in response.message
    assert (
        "//cochar" in response.message and "Chat Memory is required" in response.message
    )


def test_command_error_type_remains_compatible():
    command = parse_message("//pace invalid")[0][0]
    user = UserSettings(LocalUserStorage(), XUID("test", "salt"))
    with pytest.raises(CommandError):
        command(user, JaiRequest(), ResponseHelper())


def test_command_order_and_limits_are_atomic(run):
    user, _, _, _ = run("//rp on //rp off")
    assert not user.use_rp
    user, request, _, _ = run("//pace slow //rp off Continue.")
    assert build_prompt(user, request) == ""
    user, request, _, _ = run("//rp off //pace fast Continue.")
    assert "Move external events" in build_prompt(user, request)
    original = deepcopy(user.rp_settings)
    names = ", ".join(f"Person{i}" for i in range(17))
    user, _, response, _ = run(f"//cochars active ({names})")
    assert "At most 16" in response.message
    assert user.rp_settings == original


def test_saved_main_cast_removal_after_one_shot_clear(run):
    run("//cochar active (Pantalone)")
    user, _, _, _ = run("//cochar off this //cochar off")
    assert "cocharacters" not in user.rp_settings


def test_adult_commands_are_not_registered():
    from gfjproxy.rp_rules import COMMAND_NAMES

    assert "18+" not in COMMAND_NAMES
    assert "smut" not in COMMAND_NAMES


@pytest.mark.parametrize(
    "text, needle",
    [
        ("//rp this", "canonical_roleplay_style"),
        ("//cochar this (Pantalone)", "Pantalone"),
        ("//pace slow this", "small developments"),
        ("//scenario select active", "offer 3-5 distinct"),
    ],
)
def test_one_request_and_scenario_commands_are_useful_without_accompanying_prose(
    run, text, needle
):
    _, request, _, provider = run(text)
    provider.assert_called_once()
    assert needle in request.messages[0].content
