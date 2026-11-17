from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, get_ident
from unittest.mock import Mock

import httpx2
import pytest
from flask import Flask

from gfjproxy._globals import PROCESS_TIMEOUT
from gfjproxy.routes import proxy as proxy_route
from gfjproxy.xuiduser import XUID, LocalUserStorage, RedisUserStorage, UserSettings


@pytest.fixture(params=["local", "redis-mocked"])
def storage(request, mocker):
    if request.param == "local":
        return LocalUserStorage()

    # Exercise RedisUserStorage's request ownership with a mocked Redis server.
    # No live Redis connection is required for these concurrency regressions.
    data, mutexes = {}, {}
    client = Mock()
    client.get.side_effect = data.get
    client.exists.side_effect = lambda name: name in data

    def set_value(name, value):
        data[name] = value.encode() if isinstance(value, str) else value
        return True

    def make_lock(*, name, timeout):
        mutex = mutexes.setdefault(name, Lock())
        handle = Mock()
        handle.acquire.side_effect = lambda *, blocking: mutex.acquire(
            blocking=blocking
        )
        handle.release.side_effect = mutex.release
        return handle

    client.set.side_effect = set_value
    client.lock.side_effect = make_lock
    mocker.patch("gfjproxy.xuiduser.redis.from_url", return_value=client)
    return RedisUserStorage()


@pytest.fixture
def chat(storage, mocker, monkeypatch):
    mocker.patch.object(proxy_route, "storage", storage)
    mocker.patch.object(proxy_route, "get_cooldown", return_value=0)
    monkeypatch.setattr(proxy_route, "PROCESS_TIMEOUT", 2, raising=False)
    post = mocker.patch("gfjproxy.providers.gemini.http_client.post")
    post.return_value = httpx2.Response(
        200,
        request=httpx2.Request("POST", "https://generativelanguage.googleapis.com/"),
        json={"candidates": [{"content": {"parts": [{"text": "Bot response."}]}}]},
    )
    app = Flask(__name__)
    app.register_blueprint(proxy_route.proxy)

    def send(message="Continue.", *, key="google/test-key", stream=False):
        # Each concurrent request gets its own client and request context.
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
def test_overlapping_requests_wait_only_for_the_same_key(
    storage, chat, mocker, stream, same_key, first_status
):
    send, post = chat
    first_started, release_first, contended = Event(), Event(), Event()
    active_guard = Lock()
    active = peak_active = 0
    model_response = post.return_value
    model_error = httpx2.Response(
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
    original_lock = storage.lock

    def try_lock(xuid):
        acquired = original_lock(xuid)
        if not acquired:
            contended.set()
        return acquired

    def generate(*args, **kwargs):
        nonlocal active, peak_active
        with active_guard:
            active += 1
            peak_active = max(peak_active, active)
        try:
            text = kwargs["json"]["contents"][-1]["parts"][0]["text"]
            if text == "First.":
                first_started.set()
                assert release_first.wait(5)
                if first_status == 503:
                    return model_error
            return model_response
        finally:
            with active_guard:
                active -= 1

    mocker.patch.object(storage, "lock", side_effect=try_lock)
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
            if same_key:
                assert contended.wait(2)
            else:
                assert second.result(timeout=2).status_code == 200
        finally:
            release_first.set()
        assert first.result(timeout=3).status_code == first_status
        assert second.result(timeout=3).status_code == 200

    assert peak_active == (1 if same_key else 2)
    assert post.call_count == 2
    settings = UserSettings(storage, XUID("google/test-key", proxy_route.xuid_secret))
    assert settings.rp_settings["pace"]["value"] == "slow"
    if same_key:
        assert settings.rp_settings["focus"]["value"] == "dialogue"
        assert settings.get_rcounter() == 2


@pytest.mark.parametrize("failure", ["load", "cooldown", "handler", "save"])
def test_request_lock_is_released_even_when_processing_fails(
    storage, chat, mocker, failure
):
    send, _ = chat
    target, method = {
        "load": (storage, "get"),
        "cooldown": (proxy_route, "get_cooldown"),
        "handler": (proxy_route, "handle_chat_message"),
        "save": (storage, "put"),
    }[failure]
    if failure == "cooldown":
        mocker.patch("gfjproxy.routes.proxy.UserSettings.last_seen", return_value=1)
    patch = mocker.patch.object(
        target, method, side_effect=RuntimeError("test failure")
    )

    response = send()

    assert response.status_code == 500
    assert "Internal Proxy Error" in response.get_data(as_text=True)
    mocker.stop(patch)
    assert send().status_code == 200


@pytest.mark.parametrize("stream", [False, True])
def test_wait_timeout_does_not_release_another_requests_lock(
    storage, chat, monkeypatch, stream
):
    send, post = chat
    xuid = XUID("google/test-key", proxy_route.xuid_secret)
    assert storage.lock(xuid)
    monkeypatch.setattr(proxy_route, "PROCESS_TIMEOUT", 0)
    try:
        response = send(stream=stream)
        assert response.status_code == 504
        assert "Timed out waiting" in response.get_data(as_text=True)
        assert not storage.lock(xuid)
        post.assert_not_called()
    finally:
        storage.unlock(xuid)
    assert send(stream=stream).status_code == 200


def test_cooldown_early_return_releases_request_lock(storage, chat, mocker):
    send, post = chat
    mocker.patch("gfjproxy.routes.proxy.UserSettings.last_seen", return_value=1)
    cooldown = mocker.patch.object(proxy_route, "get_cooldown", return_value=10)
    assert send().status_code == 429
    post.assert_not_called()
    cooldown.return_value = 0
    assert send().status_code == 200


def test_provider_permission_errors_are_preserved(storage, chat):
    send, post = chat
    post.return_value = httpx2.Response(
        403,
        request=httpx2.Request("POST", "https://generativelanguage.googleapis.com/"),
        json={
            "error": {"code": 403, "status": "PERMISSION_DENIED", "message": "Denied"}
        },
    )
    response = send()
    assert response.status_code == 403
    assert "PERMISSION_DENIED" in response.get_data(as_text=True)


def test_local_waiter_cannot_acquire_a_retired_lock():
    storage = LocalUserStorage()
    xuid = XUID("test-key", "test-salt")
    owner_thread = get_ident()
    waiter_started, resume_waiter = Event(), Event()
    mutex = Lock()

    def acquire(*, blocking):
        if get_ident() != owner_thread:
            waiter_started.set()
            assert resume_waiter.wait(5)
        return mutex.acquire(blocking=blocking)

    handle = Mock()
    handle.acquire.side_effect = acquire
    handle.release.side_effect = mutex.release
    storage._locks[xuid.lockid()] = handle
    assert storage.lock(xuid)
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiter = pool.submit(storage.lock, xuid)
        try:
            assert waiter_started.wait(2)
            storage.unlock(xuid)
            assert storage.lock(xuid)
            resume_waiter.set()
            assert not waiter.result(timeout=2)
        finally:
            resume_waiter.set()
            storage.unlock(xuid)


def test_redis_expired_owner_cannot_release_the_next_owners_handle(mocker):
    client = Mock()
    first_handle, next_handle = Mock(), Mock()
    first_handle.acquire.return_value = next_handle.acquire.return_value = True
    client.lock.side_effect = [first_handle, next_handle]
    mocker.patch("gfjproxy.xuiduser.redis.from_url", return_value=client)
    storage = RedisUserStorage()
    xuid = XUID("test-key", "test-salt")
    next_acquired, release_next = Event(), Event()
    assert storage.lock(xuid)

    def next_owner():
        # Successful acquire simulates the previous owner's lease having expired.
        assert storage.lock(xuid)
        next_acquired.set()
        try:
            assert release_next.wait(5)
        finally:
            storage.unlock(xuid)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(next_owner)
        try:
            assert next_acquired.wait(2)
            storage.unlock(xuid)
            first_handle.release.assert_called_once()
            next_handle.release.assert_not_called()
        finally:
            release_next.set()
        future.result(timeout=3)
    next_handle.release.assert_called_once()
    for call in client.lock.call_args_list:
        assert call.kwargs["timeout"] > PROCESS_TIMEOUT
