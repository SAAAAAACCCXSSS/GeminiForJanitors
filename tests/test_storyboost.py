"""StoryBoost persistent and one-turn system-prompt integration tests."""

import pytest

from gfjproxy._globals import PRESETS
from gfjproxy.commands import Command, parse_message
from tests import test_roleplay

storyboost_chat = test_roleplay.chat


@pytest.mark.parametrize("mode", ["on", "off", "this"])
def test_storyboost_command_parsing(mode):
    commands, remaining = parse_message(f"//storyboost {mode} Continue.")
    assert commands == [Command("storyboost", mode)]
    assert remaining == "Continue."


@pytest.mark.parametrize("system", [True, False])
def test_storyboost_one_turn_and_persistent_modes(storyboost_chat, system):
    send, post = storyboost_chat

    def request_prompt():
        request = post.call_args.kwargs["json"]
        parts = request.get("systemInstruction", {}).get("parts", [])
        return "\n".join(part["text"] for part in parts)

    assert send("Continue.", system=system)[0] == 200
    assert PRESETS["storyboost"] not in request_prompt()

    status, confirmation = send("//storyboost this Continue.", system=system)
    assert status == 200
    assert "enabled (for this message only)" in confirmation
    assert request_prompt().count(PRESETS["storyboost"]) == 1
    if system:
        assert request_prompt().startswith("Character description.")

    send("Continue.", system=system)
    assert PRESETS["storyboost"] not in request_prompt()

    status, confirmation = send("//storyboost on Continue.", system=system)
    assert status == 200
    assert "StoryBoost enabled" in confirmation
    assert PRESETS["storyboost"] in request_prompt()

    send("Continue.", system=system)
    assert PRESETS["storyboost"] in request_prompt()

    assert "//storyboost on" in send("//aboutme", system=system)[1]

    status, confirmation = send("//storyboost off Continue.", system=system)
    assert status == 200
    assert "StoryBoost disabled" in confirmation
    assert PRESETS["storyboost"] not in request_prompt()

    send("Continue.", system=system)
    assert PRESETS["storyboost"] not in request_prompt()


def test_storyboost_command_only_persists_without_model_call(storyboost_chat):
    send, post = storyboost_chat
    status, confirmation = send("//storyboost on")
    assert status == 200
    assert "StoryBoost enabled" in confirmation
    post.assert_not_called()

    send("Continue.")
    request = post.call_args.kwargs["json"]
    assert PRESETS["storyboost"] in request["systemInstruction"]["parts"][0]["text"]


def test_storyboost_this_alone_generates_a_single_turn(storyboost_chat):
    send, post = storyboost_chat
    assert send("//storyboost this")[0] == 200
    request = post.call_args.kwargs["json"]
    assert PRESETS["storyboost"] in request["systemInstruction"]["parts"][0]["text"]
    assert request["contents"][-1]["parts"][0]["text"].strip()

    assert send("Continue.")[0] == 200
    prompt = post.call_args.kwargs["json"]["systemInstruction"]["parts"][0]["text"]
    assert PRESETS["storyboost"] not in prompt


def test_storyboost_composes_with_rp_rules(storyboost_chat):
    send, post = storyboost_chat
    assert send("//rp on //storyboost on Continue.")[0] == 200
    instruction = post.call_args.kwargs["json"]["systemInstruction"]["parts"][0]["text"]
    assert instruction.startswith("Character description.")
    assert PRESETS["canonrp"] in instruction
    assert instruction.count(PRESETS["storyboost"]) == 1


@pytest.mark.parametrize("model", ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"])
def test_storyboost_sets_real_high_thinking_for_lite(storyboost_chat, monkeypatch, model):
    """The provider receives native HIGH thinking, not a fake prose request."""
    from gfjproxy.models import JaiRequest

    original_parse = JaiRequest.parse

    def parse_with_model(payload):
        request = original_parse(payload)
        request.models = {"google": model}
        return request

    monkeypatch.setattr(JaiRequest, "parse", staticmethod(parse_with_model))
    send, post = storyboost_chat
    assert send("Continue.")[0] == 200
    assert "thinkingConfig" not in post.call_args.kwargs["json"]["generationConfig"]

    assert send("//storyboost this Continue.")[0] == 200
    generation_config = post.call_args.kwargs["json"]["generationConfig"]
    assert generation_config["thinkingConfig"] == {"thinkingLevel": "HIGH"}

    assert send("Continue.")[0] == 200
    assert "thinkingConfig" not in post.call_args.kwargs["json"]["generationConfig"]

    assert send("//storyboost on Continue.")[0] == 200
    assert post.call_args.kwargs["json"]["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "HIGH"
    }

    assert send("//storyboost off Continue.")[0] == 200
    assert "thinkingConfig" not in post.call_args.kwargs["json"]["generationConfig"]


def test_storyboost_does_not_override_other_gemini_models(storyboost_chat):
    """Unverified/other model names retain their normal reasoning defaults."""
    send, post = storyboost_chat
    assert send("//storyboost on Continue.")[0] == 200
    config = post.call_args.kwargs["json"]["generationConfig"]
    assert "thinkingConfig" not in config
