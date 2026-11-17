"""Concurrent generation without a request queue or lost settings updates."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from unittest.mock import MagicMock, Mock

import httpx2
import pytest
from flask import Flask
from redis import Redis, WatchError

from gfjproxy.roleplay import RoleplayError
from gfjproxy.routes import proxy as proxy_route
from gfjproxy.rp_rules import MODES
from gfjproxy.xuiduser import XUID, LocalUserStorage, RedisUserStorage, UserSettings


@pytest.fixture(params=["local", "redis-mocked"])
def storage(request, mocker):
    if request.param == "local":
        return LocalUserStorage()

    data, transaction_guard = {}, Lock()
    client = Mock()
    client.get.side_effect = data.get
    client.exists.side_effect = lambda name: name in data

    def set_value(name, value):
        data[name] = value.encode() if isinstance(value, str) else value
        return True

    def transaction(func, *watches, value_from_callable):
        with transaction_guard:
            pending = []
            pipe = Mock()
            pipe.get.side_effect = data.get
            pipe.set.side_effect = lambda name, value: pending.append((name, value))
            result = func(pipe)
            for name, value in pending:
                set_value(name, value)
            return result

    client.set.side_effect = set_value
    client.transaction.side_effect = transaction
    mocker.patch("gfjproxy.xuiduser.redis.from_url", return_value=client)
    return RedisUserStorage()


@pytest.fixture
def chat(storage, mocker):
    mocker.patch.object(proxy_route, "storage", storage)
    mocker.patch.object(proxy_route, "get_cooldown", return_value=0)
    # No request-wide locking or waiting is allowed, even for the same key.
    mocker.patch.object(
        storage, "lock", side_effect=AssertionError("Request lock used")
    )
    mocker.patch.object(
        storage, "unlock", side_effect=AssertionError("Request unlock used")
    )
    post = mocker.patch("gfjproxy.providers.gemini.http_client.post")
    post.return_value = httpx2.Response(
        200,
        request=httpx2.Request("POST", "https://generativelanguage.googleapis.com/"),
        json={"candidates": [{"content": {"parts": [{"text": "Bot response."}]}}]},
    )
    app = Flask(__name__)
    app.register_blueprint(proxy_route.proxy)

    def send(message="Continue.", *, key="google/test-key", stream=False):
        return app.test_client().post(
            "/chat/completions",
            headers={"Authorization": "Bearer " + key},
            json={
                "model": "google/gemini-test",
                "messages": [
                    {"role": "system", "content": "Character description."},
                    {"role": "user", "content": message},
                ],
                "stream": stream,
            },
        )

    return send, post


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("same_key", [False, True])
@pytest.mark.parametrize("first_status", [200, 503])
def test_same_key_generations_overlap_and_merge_settings(
    storage, chat, stream, same_key, first_status
):
    send, post = chat
    first_started, release_first = Event(), Event()
    active_guard = Lock()
    active = peak_active = 0
    model_response = post.return_value
    model_error = httpx2.Response(
        503,
        request=httpx2.Request("POST", "https://generativelanguage.googleapis.com/"),
        json={
            "error": {"code": 503, "status": "UNAVAILABLE", "message": "High demand"}
        },
    )
    requests = {}

    def generate(*args, **kwargs):
        nonlocal active, peak_active
        text = kwargs["json"]["contents"][-1]["parts"][0]["text"]
        requests[text] = kwargs["json"]
        with active_guard:
            active += 1
            peak_active = max(peak_active, active)
        try:
            if text == "First.":
                first_started.set()
                assert release_first.wait(5)
                if first_status == 503:
                    return model_error
            return model_response
        finally:
            with active_guard:
                active -= 1

    post.side_effect = generate
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(send, "//pace slow First.", stream=stream)
        try:
            assert first_started.wait(2)
            second = pool.submit(
                send,
                "//focus dialogue Second.",
                key="google/test-key" if same_key else "google/other-key",
                stream=stream,
            )
            # The second generation must finish while the first is still running.
            assert second.result(timeout=2).status_code == 200
            assert not first.done()
        finally:
            release_first.set()
        assert first.result(timeout=3).status_code == first_status

    assert peak_active == 2
    assert post.call_count == 2
    first_rules = requests["First."]["systemInstruction"]["parts"][0]["text"]
    second_rules = requests["Second."]["systemInstruction"]["parts"][0]["text"]
    assert MODES["pace"].options["slow"] in first_rules
    assert MODES["pace"].options["slow"] not in second_rules
    assert MODES["focus"].options["dialogue"] in second_rules
    assert MODES["focus"].options["dialogue"] not in first_rules
    settings = UserSettings(storage, XUID("google/test-key", proxy_route.xuid_secret))
    assert settings.rp_settings["pace"]["value"] == "slow"
    if same_key:
        assert settings.rp_settings["focus"]["value"] == "dialogue"
        assert settings.get_rcounter() == 2


def test_concurrent_requests_reserve_different_rotation_slots(storage, chat):
    send, post = chat
    first_started, release_first = Event(), Event()
    model_response = post.return_value

    def generate(*args, **kwargs):
        if kwargs["headers"]["x-goog-api-key"] == "test-key":
            first_started.set()
            assert release_first.wait(5)
        return model_response

    post.side_effect = generate
    keys = "google/test-key,google/second-key"
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(send, key=keys)
        try:
            assert first_started.wait(2)
            second = pool.submit(send, key=keys)
            assert second.result(timeout=2).status_code == 200
            assert not first.done()
        finally:
            release_first.set()
        assert first.result(timeout=3).status_code == 200
    assert [
        call.kwargs["headers"]["x-goog-api-key"] for call in post.call_args_list
    ] == ["test-key", "second-key"]


def test_settings_snapshots_are_independent(storage):
    xuid = XUID("test", "salt")
    first, second = UserSettings(storage, xuid), UserSettings(storage, xuid)
    first.rp_settings = {"pace": {"value": "slow"}}
    first.advsettings["temperature"] = True
    assert not second.rp_settings
    assert not storage.get(xuid)[0]
    first.save()
    assert not second.rp_settings
    second.use_search = True
    second.save()
    saved = UserSettings(storage, xuid)
    assert saved.rp_settings["pace"]["value"] == "slow"
    assert saved.advsettings["temperature"]
    assert saved.use_search


@pytest.mark.parametrize("reverse", [False, True])
def test_independent_cast_and_nested_settings_merge(storage, reverse):
    xuid = XUID("test", "salt")
    first, second = UserSettings(storage, xuid), UserSettings(storage, xuid)
    first.rp_settings = {"pace": {"value": "slow"}, "cocharacters": ["Pantalone"]}
    second.rp_settings = {
        "focus": {"value": "dialogue"},
        "cocharacters": ["Dr. Ratio", "PANTALONE"],
    }
    first.advsettings["temperature"] = True
    second.advsettings["top_p"] = True
    for user in [second, first] if reverse else [first, second]:
        user.save()
    saved = UserSettings(storage, xuid)
    assert set(saved.rp_settings) == {"pace", "focus", "cocharacters"}
    assert {name.casefold() for name in saved.rp_settings["cocharacters"]} == {
        "pantalone",
        "dr. ratio",
    }
    assert len(saved.rp_settings["cocharacters"]) == 2
    assert saved.advsettings == {"temperature": True, "top_p": True}


def test_removing_a_setting_preserves_other_concurrent_additions(storage):
    xuid = XUID("test", "salt")
    initial = UserSettings(storage, xuid)
    initial.rp_settings = {"pace": {"value": "slow"}}
    initial.save()
    first, second = UserSettings(storage, xuid), UserSettings(storage, xuid)
    first.rp_settings = {}
    second.rp_settings["focus"] = {"value": "dialogue"}
    second.save()
    first.save()
    assert UserSettings(storage, xuid).rp_settings == {"focus": {"value": "dialogue"}}


def test_explicit_disabled_toggle_is_not_lost(storage):
    xuid = XUID("test", "salt")
    first, second = UserSettings(storage, xuid), UserSettings(storage, xuid)
    first.use_rp = True
    second.use_rp = False
    first.save()
    second.save()
    assert not UserSettings(storage, xuid).use_rp


def test_merged_cast_still_respects_limits(storage):
    xuid = XUID("test", "salt")
    first, second = UserSettings(storage, xuid), UserSettings(storage, xuid)
    first.rp_settings = {"cocharacters": [f"Hero {i}" for i in range(15)]}
    second.rp_settings = {"cocharacters": ["Another", "One more"]}
    first.save()
    with pytest.raises(RoleplayError, match="At most"):
        second.save()
    assert len(UserSettings(storage, xuid).rp_settings["cocharacters"]) == 15


@pytest.mark.parametrize("failure", ["load", "counter", "handler", "save"])
def test_internal_errors_do_not_block_following_requests(
    storage, chat, mocker, failure
):
    send, _ = chat
    target, method = {
        "load": (storage, "get"),
        "counter": (storage, "update"),
        "handler": (proxy_route, "handle_chat_message"),
        "save": (UserSettings, "save"),
    }[failure]
    patch = mocker.patch.object(
        target, method, side_effect=RuntimeError("test failure")
    )
    assert send().status_code == 500
    mocker.stop(patch)
    assert send().status_code == 200


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "message", ["//rp invalid Continue.", "//think invalid Continue."]
)
def test_command_errors_are_http_errors_without_generation(chat, message, stream):
    send, post = chat
    response = send(message, stream=stream)
    assert response.status_code == 400
    assert response.mimetype == "text/plain"
    assert "Error:" in response.get_data(as_text=True)
    assert '"choices"' not in response.get_data(as_text=True)
    post.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 503, 504])
def test_provider_errors_are_http_errors(chat, status, stream):
    send, post = chat
    post.return_value = httpx2.Response(
        status,
        request=httpx2.Request("POST", "https://generativelanguage.googleapis.com/"),
        json={
            "error": {"code": status, "status": "TEST_ERROR", "message": "Test failure"}
        },
    )
    response = send("//rp on Continue.", stream=stream)
    assert response.status_code == status
    assert response.mimetype == "text/plain"
    text = response.get_data(as_text=True)
    assert "Test failure" in text
    assert "<proxy>" not in text and '"choices"' not in text
    assert not text.startswith("data: ")


def test_redis_watch_conflict_retries_against_fresh_settings(mocker):
    xuid = XUID("test", "salt")
    client = Mock()
    client.get.return_value = None
    pipe = MagicMock()
    client.pipeline.return_value = MagicMock()
    client.pipeline.return_value.__enter__.return_value = pipe
    pipe.get.side_effect = [None, json.dumps({"use_search": True})]
    pipe.execute.side_effect = [WatchError(), [True]]
    client.transaction.side_effect = lambda func, *keys, **kwargs: Redis.transaction(
        client, func, *keys, **kwargs
    )
    mocker.patch("gfjproxy.xuiduser.redis.from_url", return_value=client)
    storage = RedisUserStorage()
    user = UserSettings(storage, xuid)
    user.use_rp = True
    user.save()
    assert pipe.execute.call_count == 2
    saved = json.loads(pipe.set.call_args.args[1])
    assert saved["use_rp"] and saved["use_search"]
    assert user.use_search
