import httpx
import pytest

from elice_rag.config import Settings, parse_mapping
from elice_rag.providers.elice import (
    EliceConfigError,
    _extract_chat_delta,
    _extract_chat_text,
    _post_json_with_retry,
    _resolve_chat_url,
    _resolve_embedding_url,
    _select_chat_endpoint,
)


def test_elice_endpoint_root_uses_chat_completions_path():
    url = "https://mlapi.run/example-endpoint-id"

    assert _resolve_chat_url(url) == f"{url}/v1/chat/completions"


def test_elice_openai_compatible_base_uses_chat_completions_path():
    url = "https://mlapi.run/example-endpoint-id/v1"

    assert _resolve_chat_url(url) == f"{url}/chat/completions"


def test_elice_openai_compatible_base_uses_embeddings_path():
    url = "https://mlapi.run/example-endpoint-id/v1"

    assert _resolve_embedding_url(url) == f"{url}/embeddings"


def test_elice_endpoint_root_uses_embeddings_path():
    url = "https://mlapi.run/example-embedding-endpoint-id"

    assert _resolve_embedding_url(url) == f"{url}/v1/embeddings"


def test_elice_embedding_url_accepts_full_chat_completions_path():
    url = "https://mlapi.run/example-endpoint-id/v1/chat/completions"

    assert _resolve_embedding_url(url) == "https://mlapi.run/example-endpoint-id/v1/embeddings"


def test_elice_chat_response_parses_openai_and_direct_shapes():
    openai_shape = {"choices": [{"message": {"content": "answer"}}]}
    direct_shape = {"result": "answer"}

    assert _extract_chat_text(openai_shape) == "answer"
    assert _extract_chat_text(direct_shape) == "answer"


def test_elice_stream_delta_parses_openai_shape():
    chunk = {"choices": [{"delta": {"content": "partial answer"}}]}

    assert _extract_chat_delta(chunk) == "partial answer"


def test_elice_chat_response_requires_supported_text_field():
    with pytest.raises(EliceConfigError):
        _extract_chat_text({"unexpected": "shape"})


def test_parse_elice_model_endpoint_mapping():
    mapping = parse_mapping(
        "openai/gpt-5-mini=https://mlapi.run/mini/v1; "
        "openai/gpt-5-nano=https://mlapi.run/nano/v1"
    )

    assert mapping == {
        "openai/gpt-5-mini": "https://mlapi.run/mini/v1",
        "openai/gpt-5-nano": "https://mlapi.run/nano/v1",
    }


def test_elice_model_mapping_selects_requested_endpoint():
    settings = Settings(
        elice_base_url="https://mlapi.run/mini/v1",
        elice_chat_model="openai/gpt-5-mini",
        elice_model_endpoints={
            "openai/gpt-5-mini": "https://mlapi.run/mini/v1",
            "openai/gpt-5-nano": "https://mlapi.run/nano/v1",
        },
    )

    assert _select_chat_endpoint(settings, "openai/gpt-5-nano") == "https://mlapi.run/nano/v1"


def test_elice_endpoint_root_allows_provider_to_validate_requested_model():
    settings = Settings(
        elice_base_url="https://mlapi.run/mini",
        elice_chat_model="openai/gpt-5-mini",
    )

    assert _select_chat_endpoint(settings, "openai/gpt-5-nano") == "https://mlapi.run/mini"


def test_elice_post_json_retries_transient_gateway_error():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(502, json={"error": "temporary gateway error"})
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        response = _post_json_with_retry(
            client,
            "https://mlapi.run/example/v1/chat/completions",
            headers={},
            payload={"messages": []},
            base_sleep_seconds=0,
        )

    assert response.status_code == 200
    assert calls == 2
