import pytest

from gfjproxy.utils import ResponseHelper, comma_split, is_proxy_test

################################################################################

SAMPLE_NORMAL_CHAT_JSON = {
    "messages": [
        {
            "content": "Bot description.\n<UserPersona>User persona description</UserPersona>\n<example_dialogs>Blah blah blah</example_dialogs>\n",
            "role": "system",
        },
        {"content": ".", "role": "user"},
        {"content": "Bot initial message", "role": "assistant"},
        {"content": "User persona: inital message", "role": "user"},
        {"content": "Bot reply", "role": "assistant"},
    ],
    "model": "gemini-2.5-pro",
    "stream": False,
    "temperature": 0.8,
}

SAMPLE_PROXY_TEST_JSON = {
    "max_tokens": 10,
    "messages": [{"content": "Just say TEST", "role": "user"}],
    "model": "gemini-2.5-pro",
    "temperature": 0,
}


def test_is_proxy_test():
    """Basic parsing tests."""

    assert is_proxy_test(SAMPLE_PROXY_TEST_JSON)

    assert not is_proxy_test(SAMPLE_NORMAL_CHAT_JSON)


################################################################################


def test_comma_split():
    """Basic comma split algorithm tests."""

    assert comma_split("A") == ["A"]
    assert comma_split("A, B, C") == ["A", "B", "C"]

    # Drop trailing comma(s)
    assert comma_split("A, B, C,") == ["A", "B", "C"]
    assert comma_split("A, B, C,,") == ["A", "B", "C"]

    # Drop leading comma(s)
    assert comma_split(", A, B, C") == ["A", "B", "C"]
    assert comma_split(",, A, B, C") == ["A", "B", "C"]

    # Drop extraneous comma(s)
    assert comma_split("A,, B, C") == ["A", "B", "C"]
    assert comma_split("A,, B,, C") == ["A", "B", "C"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("wrap_errors", [False, True])
@pytest.mark.parametrize("status", [400, 429, 503, 504])
def test_errors_with_command_output_preserve_http_status(stream, wrap_errors, status):
    response = (
        ResponseHelper(use_stream=stream, wrap_errors=wrap_errors)
        .add_proxy_message("Setting saved.")
        .add_error(f"Error from provider ({status}): unavailable", status)
        .add_proxy_message("Additional diagnostic details.")
        .build()
    )

    assert response.status_code == status
    expected = (
        f"Setting saved.\nError from provider ({status}): unavailable\n"
        "Additional diagnostic details."
    )
    if wrap_errors:
        assert response.mimetype == "application/json"
        assert response.get_json() == {"error": expected}
    else:
        assert response.mimetype == "text/plain"
        assert response.get_data(as_text=True) == expected


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("wrap_errors", [False, True])
def test_error_with_diagnostics_only_remains_an_error(stream, wrap_errors):
    response = (
        ResponseHelper(use_stream=stream, wrap_errors=wrap_errors)
        .add_error("Unavailable", 503)
        .add_proxy_message("Try again later.")
        .build()
    )
    assert response.status_code == 503
    expected = "Error 503: Unavailable\nTry again later."
    if wrap_errors:
        assert response.get_json() == {"error": expected}
    else:
        assert response.get_data(as_text=True) == expected


################################################################################


def test_response_helper():
    """Usage test."""

    def make_jai_res_str(*content):
        content = repr("".join(content))[1:-1]
        return [
            '{"choices": [{"index": 0, "message": {"role": "assistant", "content": "'
            + content
            + '"}, "finish_reason": "stop"}]}'
        ]

    new_line = "\n"
    proxy_open = ResponseHelper.PROXY_TAG_OPEN
    proxy_close = ResponseHelper.PROXY_TAG_CLOSE

    # Basic error responses (Google AI or something returned an error)
    # ----------------------------------------------------------------

    assert (  # Ensure the status_code is preserved
        ResponseHelper().build_error("Lorem Ipsum", 400).status_code == 400
    )

    assert (  # Chat response error (JanitorAI will prepend "PROXY ERROR")
        ResponseHelper().build_error("Lorem Ipsum", 400).response == ["Lorem Ipsum"]
    )

    assert (  # Proxy test error (JanitorAI shows this error message unaltered)
        ResponseHelper(wrap_errors=True)
        .build_error("Lorem Ipsum", 400)
        .response  # flask.Response.response
        == ['{"error": "PROXY ERROR 400: Lorem Ipsum"}']
    )

    assert (  # Multiple error messages preserve the final error status
        ResponseHelper()
        .add_error("Lorem Ipsum", 400)
        .add_error("Dolor Sit", 300)
        .build()
        .status_code
        == 300
    )

    assert (  # Error responses contain diagnostics without chat markup
        ResponseHelper()
        .add_error("Lorem", 400)
        .add_error("Ipsum", 300)
        .build()
        .response  # flask.Response.response
        == ["Error 400: Lorem\nError 300: Ipsum"]
    )

    # Basic chat responses (successful responses from Google AI)
    # ----------------------------------------------------------

    assert (  # Ensure the status_code is 200 for JanitorAI to accept the message
        ResponseHelper().build_message("Hello, World!").status_code == 200
    )

    assert (  # Ensure the model's output is unaltered in this case
        ResponseHelper().add_message("Hello, World!").message == "Hello, World!"
    )

    assert (  # Don't strip white space in this case
        ResponseHelper().add_message("  \nZXY  \n").message == "  \nZXY  \n"
    )

    assert (  # Ensure the result is in the valid JanitorAI response format
        ResponseHelper().build_message("abcdef").response == make_jai_res_str("abcdef")
    )

    assert (  # wrap_errors should NOT have an effect on this
        ResponseHelper(wrap_errors=True).build_message("abcdef").response
        == make_jai_res_str("abcdef")
    )

    assert (  # Ensure support for joining multiple bot messages together
        ResponseHelper().add_message("Hello, World!", "Lorem Ipsum").message
        == "Hello, World!\nLorem Ipsum"
    )

    assert (  # Multiple bot messages should not affect the final result
        ResponseHelper()
        .build_message("Hello, World!", "Lorem Ipsum")
        .response  # flask.Response.response
        == make_jai_res_str("Hello, World!", new_line, "Lorem Ipsum")
    )

    # Basic proxy chat responses (non-inline command output from proxy)
    # -----------------------------------------------------------------

    assert (  # Ensure the status_code is 200 for JanitorAI to accept the message
        ResponseHelper().add_proxy_message("Hello, World!").build().status_code == 200
    )

    assert (  # Ensure a message is wrapped within <proxy> tags
        ResponseHelper().add_proxy_message("Hello, World!").message
        == f"{proxy_open}Hello, World!{proxy_close}"
    )

    assert (  # Join multiple proxy messages together
        ResponseHelper()
        .add_proxy_message("ABC")
        .add_proxy_message("DEF")
        .add_proxy_message("GHI")
        .message
        == f"{proxy_open}ABC\nDEF\nGHI{proxy_close}"
    )

    assert (  # Ensure the result is in the valid JanitorAI response format
        ResponseHelper()
        .add_proxy_message("abc")
        .add_proxy_message("def")
        .add_proxy_message("ghi")
        .build()
        .response  # flask.Response.response
        == make_jai_res_str(
            proxy_open, "abc", new_line, "def", new_line, "ghi", proxy_close
        )
    )

    # Complex multi-line chat responses (command output and bot responses)
    # --------------------------------------------------------------------

    assert (  # Ensure a proxy message in prefix position works
        ResponseHelper().add_proxy_message("ABC").add_message("DEF").message
        == f"{proxy_open}ABC{proxy_close}\nDEF"
    )

    assert (  # Ensure a proxy message in suffix position works
        ResponseHelper().add_message("ABC").add_proxy_message("DEF").message
        == f"ABC\n{proxy_open}DEF{proxy_close}"
    )

    assert (  # Ensure a proxy message can be sandwiched between bot responses
        ResponseHelper()
        .add_message("ABC")
        .add_proxy_message("DEF")
        .add_message("GHI")
        .message
        == f"ABC\n{proxy_open}DEF{proxy_close}\nGHI"
    )

    assert (  # Ensure proxy messages coalesce
        ResponseHelper()
        .add_message("ABC")
        .add_proxy_message("LOREM")
        .add_proxy_message("IPSUM")
        .add_message("DEF")
        .message
        == f"ABC\n{proxy_open}LOREM\nIPSUM{proxy_close}\nDEF"
    )

    assert (  # Ensure proxy messages can be interleaved with bot responses
        ResponseHelper()
        .add_message("AAA")
        .add_proxy_message("BBB")
        .add_message("CCC")
        .add_proxy_message("DDD")
        .add_message("EEE")
        .message
        == new_line.join(
            [
                "AAA",
                f"{proxy_open}BBB{proxy_close}",
                "CCC",
                f"{proxy_open}DDD{proxy_close}",
                "EEE",
            ]
        )
    )

    # Complex multi-line chat responses (chat, error and proxy)
    # ---------------------------------------------------------

    assert (  # Ensure errors are wrapped like proxy messages
        ResponseHelper()
        .add_message("AAA")
        .add_error("BBB", 404)
        .add_message("CCC")
        .message
        == new_line.join(
            [
                "AAA",
                f"{proxy_open}Error 404: BBB{proxy_close}",
                "CCC",
            ]
        )
    )

    assert (  # Ensure errors and proxy coalesce together
        ResponseHelper()
        .add_message("AAA")
        .add_error("BBB", 404)
        .add_proxy_message("CCC")
        .add_message("DDD")
        .message
        == new_line.join(
            [
                "AAA",
                f"{proxy_open}Error 404: BBB\nCCC{proxy_close}",
                "DDD",
            ]
        )
    )

    response = (  # Mixed output must still be returned as a failed request
        ResponseHelper()
        .add_message("  Bot reply with leading spaces")
        .add_proxy_message("Proxy banner")
        .add_error("Proxy had an error", 500)
        .add_message("This comes after the closing tag")
        .add_message("Bot reply with trailing spaces  ")
        .add_proxy_message("More proxy banner")
        .add_error("Proxy had another error", 502)
        .add_message("Trailing bot reply")
        .build()
    )
    assert response.status_code == 502
    assert response.mimetype == "text/plain"
    assert response.get_data(as_text=True) == new_line.join(
        [
            "  Bot reply with leading spaces",
            "Proxy banner",
            "Error 500: Proxy had an error",
            "This comes after the closing tag",
            "Bot reply with trailing spaces  ",
            "More proxy banner",
            "Error 502: Proxy had another error",
            "Trailing bot reply",
        ]
    )


################################################################################
