"""Tests for server helper behavior and app wiring."""

import builtins

import pytest

from tinkuy import server
from tinkuy import gateway as gw_mod


def test_find_free_port_returns_int():
    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def bind(self, _addr):
            return None

        def getsockname(self):
            return ("127.0.0.1", 54321)

    def fake_socket(*_args, **_kwargs):
        return FakeSocket()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(server.socket, "socket", fake_socket)
    try:
        port = server.find_free_port()
    finally:
        monkeypatch.undo()

    assert isinstance(port, int)
    assert 0 < port < 65536


def test_extract_user_content_parses_simple_text_messages():
    messages = [
        {"role": "assistant", "content": "earlier"},
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ]

    user_content, tool_results = gw_mod._extract_user_content(messages)

    assert user_content == "first\nsecond"
    assert tool_results is None


def test_extract_user_content_parses_content_blocks_and_tool_results():
    messages = [
        {"role": "assistant", "content": "earlier"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "please use tool"},
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": [
                        {"type": "text", "text": "alpha"},
                        {"type": "text", "text": "beta"},
                    ],
                },
            ],
        },
    ]

    user_content, tool_results = gw_mod._extract_user_content(messages)

    assert user_content == "please use tool"
    assert tool_results == [{
        "content": "alpha beta",
        "name": "toolu_123",
        "metadata": {"tool_use_id": "toolu_123", "is_error": False},
    }]


def test_extract_response_text_reads_anthropic_text_blocks():
    resp_data = {
        "id": "msg_1",
        "content": [
            {"type": "text", "text": "line 1"},
            {"type": "tool_use", "id": "t1", "name": "search", "input": {}},
            {"type": "text", "text": "line 2"},
            "ignored",
        ],
    }

    text, blocks = gw_mod._extract_response_content_from_json(resp_data)
    assert text == "line 1\nline 2"
    assert len(blocks) == 3  # 2 text + 1 tool_use


def test_create_app_returns_fastapi_with_expected_routes_and_health():
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    app = server.create_app()

    assert isinstance(app, fastapi.FastAPI)

    route_methods = {
        route.path: route.methods
        for route in app.routes
        if hasattr(route, "methods")
    }
    assert "/v1/messages" in route_methods
    assert "POST" in route_methods["/v1/messages"]
    assert "/v1beta/models/{model_id}:streamGenerateContent" in route_methods
    assert "POST" in route_methods["/v1beta/models/{model_id}:streamGenerateContent"]
    assert "/v1/tinkuy/status" in route_methods
    assert "GET" in route_methods["/v1/tinkuy/status"]
    assert "/v1/tinkuy/health" in route_methods
    assert "GET" in route_methods["/v1/tinkuy/health"]

    client = TestClient(app)
    client.close()


def test_serve_raises_importerror_when_uvicorn_unavailable(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "uvicorn":
            raise ImportError("No module named uvicorn")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(
        ImportError,
        match=r"Serving requires extra dependencies\. Install with: pip install tinkuy\[serve\]",
    ):
        server.serve(port=8340)
