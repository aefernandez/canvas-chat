import httpx
import litellm
from fastapi.testclient import TestClient

import canvas_chat.app as app_module
from canvas_chat.app import app
from canvas_chat.config import AppConfig


def test_provider_models_copilot_without_api_key(monkeypatch):
    """Copilot models should be available without api_key in request."""
    monkeypatch.setattr(
        litellm,
        "github_copilot_models",
        {"github_copilot/gpt-4o", "github_copilot/gpt-4o-mini"},
    )

    client = TestClient(app)
    response = client.post("/api/provider-models", json={"provider": "github_copilot"})

    assert response.status_code == 200
    data = response.json()
    assert data, "Expected copilot models in response"
    ids = {model["id"] for model in data}
    assert "github_copilot/gpt-4o" in ids
    assert all(model["provider"] == "GitHub Copilot" for model in data)


def test_provider_models_copilot_blocked_in_admin_mode(monkeypatch):
    """Copilot models should be blocked in admin mode."""
    admin_config = AppConfig(models=[], plugins=[], admin_mode=True)
    monkeypatch.setattr(app_module, "get_admin_config", lambda: admin_config)

    client = TestClient(app)
    response = client.post("/api/provider-models", json={"provider": "github_copilot"})

    assert response.status_code == 400
    assert "admin mode" in response.json()["detail"].lower()


def _mock_anthropic(monkeypatch, handler):
    """Route the app's httpx.AsyncClient through a mock transport."""
    real_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(app_module.httpx, "AsyncClient", client_factory)


def _anthropic_model(model_id, created_at, max_input_tokens=200000):
    return {
        "type": "model",
        "id": model_id,
        "display_name": model_id.replace("-", " ").title(),
        "created_at": created_at,
        "max_input_tokens": max_input_tokens,
    }


def test_provider_models_anthropic_fetches_live_list(monkeypatch):
    """Anthropic models come from /v1/models, paginated and sorted newest first."""
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["x-api-key"] == "sk-ant-test"
        assert request.headers["anthropic-version"] == "2023-06-01"
        if "after_id" not in request.url.params:
            return httpx.Response(
                200,
                json={
                    "data": [
                        _anthropic_model("claude-old", "2024-01-01T00:00:00Z"),
                    ],
                    "has_more": True,
                    "last_id": "claude-old",
                },
            )
        assert request.url.params["after_id"] == "claude-old"
        return httpx.Response(
            200,
            json={
                "data": [
                    _anthropic_model(
                        "claude-new", "2026-01-01T00:00:00Z", max_input_tokens=1000000
                    ),
                ],
                "has_more": False,
                "last_id": "claude-new",
            },
        )

    _mock_anthropic(monkeypatch, handler)

    client = TestClient(app)
    response = client.post(
        "/api/provider-models",
        json={"provider": "anthropic", "api_key": "sk-ant-test"},
    )

    assert response.status_code == 200
    assert len(requests) == 2
    data = response.json()
    assert [m["id"] for m in data] == ["anthropic/claude-new", "anthropic/claude-old"]
    assert data[0]["name"] == "Claude New"
    assert data[0]["context_window"] == 1000000
    assert all(m["provider"] == "Anthropic" for m in data)


def test_provider_models_anthropic_missing_context_window_uses_fallback(monkeypatch):
    """Models without max_input_tokens fall back to the known context windows."""
    model = _anthropic_model("claude-x", "2026-01-01T00:00:00Z")
    del model["max_input_tokens"]

    _mock_anthropic(
        monkeypatch,
        lambda request: httpx.Response(200, json={"data": [model], "has_more": False}),
    )

    client = TestClient(app)
    response = client.post(
        "/api/provider-models",
        json={"provider": "anthropic", "api_key": "sk-ant-test"},
    )

    assert response.status_code == 200
    assert response.json()[0]["context_window"] == 200000


def test_provider_models_anthropic_invalid_key(monkeypatch):
    """A rejected key returns 401 instead of a model list."""
    _mock_anthropic(monkeypatch, lambda request: httpx.Response(401, json={}))

    client = TestClient(app)
    response = client.post(
        "/api/provider-models",
        json={"provider": "anthropic", "api_key": "sk-ant-bad"},
    )

    assert response.status_code == 401


def test_provider_models_anthropic_unreachable(monkeypatch):
    """Network failures surface as 502 rather than an empty model list."""

    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    _mock_anthropic(monkeypatch, handler)

    client = TestClient(app)
    response = client.post(
        "/api/provider-models",
        json={"provider": "anthropic", "api_key": "sk-ant-test"},
    )

    assert response.status_code == 502
