import httpx2
import pytest

from gfjproxy.models import JaiMessage
from gfjproxy.providers.gemini import gemini_generate_content
from gfjproxy.xuiduser import XUID


@pytest.fixture
def post(mocker):
    post = mocker.patch("gfjproxy.providers.gemini.http_client.post")
    post.return_value = httpx2.Response(
        200,
        request=httpx2.Request("POST", "https://generativelanguage.googleapis.com/"),
        json={"candidates": [{"content": {"parts": [{"text": "Reply"}]}}]},
    )
    return post


def generate(messages):
    result = gemini_generate_content(
        XUID("test", "salt"),
        "test-key",
        "gemini-test",
        messages,
        {"temperature": 0.8, "max_tokens": 1024, "search": True},
    )
    assert result.text == "Reply"


@pytest.mark.parametrize(
    "system_texts", [[], [""], ["  "], ["Character."], ["Character.", "Style."]]
)
def test_system_instructions_are_separate_from_chat(post, monkeypatch, system_texts):
    monkeypatch.delenv("GFJPROXY_SYSTEM_INSTRUCTION", raising=False)
    messages = [JaiMessage(role="system", content=text) for text in system_texts]
    messages += [
        JaiMessage(role="user", content="Hello"),
        JaiMessage(role="assistant", content="Welcome"),
        JaiMessage(role="user", content="Continue"),
    ]
    generate(messages)
    request = post.call_args.kwargs["json"]
    assert request["contents"] == [
        {"role": "user", "parts": [{"text": "Hello"}]},
        {"role": "model", "parts": [{"text": "Welcome"}]},
        {"role": "user", "parts": [{"text": "Continue"}]},
    ]
    nonempty = [text for text in system_texts if text.strip()]
    if nonempty:
        assert request["systemInstruction"] == {
            "parts": [{"text": "\n\n".join(nonempty)}]
        }
    else:
        assert "systemInstruction" not in request
    assert request["generationConfig"] == {"temperature": 0.8, "maxOutputTokens": 1024}
    assert request["tools"] == [{"googleSearch": {}}]
    assert [
        message.content for message in messages[: len(system_texts)]
    ] == system_texts


@pytest.mark.parametrize("value", ["0", "false", "off", "no", " OFF "])
def test_system_instructions_can_use_legacy_format(post, monkeypatch, value):
    monkeypatch.setenv("GFJPROXY_SYSTEM_INSTRUCTION", value)
    generate(
        [
            JaiMessage(role="system", content="Character."),
            JaiMessage(role="user", content="Hello"),
        ]
    )
    request = post.call_args.kwargs["json"]
    assert "systemInstruction" not in request
    assert request["contents"] == [
        {"role": "model", "parts": [{"text": "Character."}]},
        {"role": "user", "parts": [{"text": "Hello"}]},
    ]
