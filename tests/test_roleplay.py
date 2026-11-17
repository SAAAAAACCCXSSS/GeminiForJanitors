import json

import httpx2
import pytest
from flask import Flask

from gfjproxy._globals import BANNER, PRESETS
from gfjproxy.routes import proxy as proxy_route
from gfjproxy.xuiduser import LocalUserStorage


@pytest.fixture
def chat(mocker, monkeypatch):
    monkeypatch.delenv("GFJPROXY_SYSTEM_INSTRUCTION", raising=False)
    storage = LocalUserStorage()
    storage.announcement = "AUTOMATIC ADMIN ANNOUNCEMENT"
    mocker.patch.object(proxy_route, "storage", storage)
    mocker.patch.object(proxy_route, "get_cooldown", return_value=0)
    post = mocker.patch("gfjproxy.providers.gemini.http_client.post")
    post.return_value = httpx2.Response(
        200,
        request=httpx2.Request("POST", "https://generativelanguage.googleapis.com/"),
        json={"candidates": [{"content": {"parts": [{"text": "Bot response."}]}}]},
    )
    app = Flask(__name__)
    app.register_blueprint(proxy_route.proxy)
    client = app.test_client()

    def send(message="Continue.", *, path="/", stream=False, system=True):
        messages = [{"role": "user", "content": message}]
        if system:
            messages.insert(0, {"role": "system", "content": "Character description."})
        response = client.post(
            path,
            headers={"Authorization": "Bearer google/test-key"},
            json={
                "model": "google/gemini-test",
                "messages": messages,
                "stream": stream,
            },
        )
        if response.status_code != 200:
            text = response.get_data(as_text=True)
        elif stream:
            data = (
                response.get_data(as_text=True).splitlines()[0].removeprefix("data: ")
            )
            text = json.loads(data)["choices"][0]["delta"]["content"]
        else:
            text = response.get_json()["choices"][0]["message"]["content"]
        return response.status_code, text

    return send, post


@pytest.mark.parametrize(
    "path", ["/", "/chat/completions", "/quiet/", "/quiet/chat/completions"]
)
@pytest.mark.parametrize("stream", [False, True])
def test_chat_has_no_automatic_notices(chat, path, stream):
    send, _ = chat
    assert send(path=path, stream=stream) == (200, "Bot response.")
    # A returning user should also see only the model's response.
    assert send(path=path, stream=stream) == (200, "Bot response.")


@pytest.mark.parametrize("stream", [False, True])
def test_commands_and_errors_stay_visible(chat, stream):
    send, post = chat
    assert "Thinking enabled" in send("//think on Continue.", stream=stream)[1]
    assert "only accepts" in send("//rp invalid Continue.", stream=stream)[1]
    assert BANNER in send("//banner", stream=stream)[1]
    post.side_effect = httpx2.ReadTimeout("test timeout")
    assert "Gateway Timeout" in send(stream=stream)[1]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("path", ["/chat/completions", "/quiet/chat/completions"])
def test_provider_503_after_command_is_an_http_error(chat, stream, path):
    send, post = chat
    send("//rp on")
    post.return_value = httpx2.Response(
        503,
        request=httpx2.Request("POST", "https://generativelanguage.googleapis.com/"),
        json={
            "error": {
                "code": 503,
                "status": "UNAVAILABLE",
                "message": "This model is currently experiencing high demand.",
            }
        },
    )

    status, text = send("//rp off Continue.", stream=stream, path=path)

    assert status == 503
    assert "Canonical RP mode disabled" in text
    assert "Saved for the primary API key" in text
    assert "Error from Google AI (503): UNAVAILABLE" in text
    assert "This model is currently experiencing high demand." in text
    assert "<proxy>" not in text
    assert '"choices"' not in text
    assert not text.startswith("data: ")
    post.assert_called_once()
    # The command was saved even though the subsequent generation failed.
    post.reset_mock()
    assert "saved RP: off" in send("//rp status")[1]
    post.assert_not_called()


@pytest.mark.parametrize("system", [False, True])
def test_rp_settings_and_prompt_survive_across_requests(chat, system):
    send, post = chat

    def has_rp():
        request = post.call_args.kwargs["json"]
        instruction = request.get("systemInstruction", {}).get("parts", [])
        text = "\n".join(part["text"] for part in instruction)
        if system:
            assert text.startswith("Character description.")
        return PRESETS["canonrp"] in text

    assert send(system=system)[0] == 200
    assert not has_rp()
    assert (
        "enabled (for this message only)"
        in send("//rp this Continue.", system=system)[1]
    )
    assert has_rp()
    send(system=system)
    assert not has_rp()
    assert "enabled." in send("//rp on Continue.", system=system)[1]
    assert has_rp()
    send(system=system)
    assert has_rp()
    assert "//rp on" in send("//aboutme", system=system)[1]
    assert "disabled." in send("//rp off Continue.", system=system)[1]
    assert not has_rp()
    send(system=system)
    assert not has_rp()
    assert "//rp off" in send("//aboutme", system=system)[1]


def test_rp_preserves_nobot_behavior(chat):
    send, post = chat
    send("//rp this //nobot this Continue.")
    instruction = post.call_args.kwargs["json"]["systemInstruction"]
    assert instruction == {"parts": [{"text": PRESETS["canonrp"]}]}


def test_rp_help(chat):
    send, _ = chat
    assert "//rp on|off|this" in send("//help commands")[1]


@pytest.mark.parametrize("stream", [False, True])
def test_extended_rp_commands_work_through_http_without_chat_memory(chat, stream):
    send, post = chat
    assert "enabled" in send("//cochar active (Pantalone)", stream=stream)[1]
    post.assert_not_called()
    status, text = send("//love hidden [Pantalone] Continue.", stream=stream)
    assert status == 200 and "Bot response." in text
    request = post.call_args.kwargs["json"]
    instruction = request["systemInstruction"]["parts"][0]["text"]
    assert "MAIN CAST" in instruction and "Pantalone" in instruction
    assert "Character [Pantalone]: Portray restrained" in instruction
    assert request["contents"][-1]["parts"][0]["text"] == "Continue."
    post.reset_mock()
    assert "Main cast: Pantalone" in send("//rp status", stream=stream)[1]
    post.assert_not_called()
    assert "reset" in send("//rp reset", stream=stream)[1]
    post.assert_not_called()
