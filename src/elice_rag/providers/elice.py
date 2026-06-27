from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from elice_rag.config import Settings


class EliceConfigError(RuntimeError):
    pass


@dataclass
class EliceChatClient:
    settings: Settings
    provider_name: str = "elice"

    @property
    def model_name(self) -> str:
        return self.settings.elice_chat_model

    def _validate(self) -> None:
        missing = [
            name
            for name, value in {
                "ELICE_API_KEY": self.settings.elice_api_key,
                "ELICE_BASE_URL": self.settings.elice_base_url,
                "ELICE_CHAT_MODEL": self.settings.elice_chat_model,
            }.items()
            if not value
        ]
        if missing:
            raise EliceConfigError("Missing Elice settings: " + ", ".join(missing))

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str:
        self._validate()
        selected_model = model or self.settings.elice_chat_model
        endpoint = _select_chat_endpoint(self.settings, selected_model)
        url = _resolve_chat_url(endpoint)
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.settings.elice_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1600,
        }
        if _uses_openai_compatible_chat_path(endpoint):
            payload["model"] = selected_model
        with httpx.Client(timeout=60) as client:
            response = _post_chat_with_compat(
                client,
                url,
                headers=headers,
                payload=payload,
            )
            response.raise_for_status()
        data = response.json()
        return _extract_chat_text(data)

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> Iterator[str]:
        self._validate()
        selected_model = model or self.settings.elice_chat_model
        endpoint = _select_chat_endpoint(self.settings, selected_model)
        url = _resolve_chat_url(endpoint)
        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self.settings.elice_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "messages": messages,
            "max_completion_tokens": 1600,
            "stream": True,
        }
        if _uses_openai_compatible_chat_path(endpoint):
            payload["model"] = selected_model
        with httpx.Client(timeout=120) as client:
            with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    text = raw_line.strip()
                    if not text or text.startswith(":"):
                        continue
                    if not text.startswith("data:"):
                        continue
                    data_text = text.removeprefix("data:").strip()
                    if data_text == "[DONE]":
                        break
                    try:
                        data = json.loads(data_text)
                    except json.JSONDecodeError:
                        continue
                    delta = _extract_chat_delta(data)
                    if delta:
                        yield delta


@dataclass
class EliceEmbeddingClient:
    settings: Settings
    provider_name: str = "elice"

    @property
    def model_name(self) -> str:
        return self.settings.elice_embedding_model

    def _validate(self) -> None:
        embedding_base_url = (
            self.settings.elice_embedding_base_url or self.settings.elice_base_url
        )
        missing = [
            name
            for name, value in {
                "ELICE_API_KEY": self.settings.elice_api_key,
                "ELICE_EMBEDDING_BASE_URL or ELICE_BASE_URL": embedding_base_url,
                "ELICE_EMBEDDING_MODEL": self.settings.elice_embedding_model,
            }.items()
            if not value
        ]
        if missing:
            raise EliceConfigError("Missing Elice embedding settings: " + ", ".join(missing))

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._validate()
        endpoint = self.settings.elice_embedding_base_url or self.settings.elice_base_url
        url = _resolve_embedding_url(endpoint)
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.settings.elice_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.elice_embedding_model,
            "input": texts,
        }
        with httpx.Client(timeout=60) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]


def _resolve_chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    if _looks_like_elice_endpoint_root(base):
        return f"{base}/v1/chat/completions"
    return base


def _select_chat_endpoint(settings: Settings, model: str) -> str:
    if model in settings.elice_model_endpoints:
        return settings.elice_model_endpoints[model]
    if model == settings.elice_chat_model:
        return settings.elice_base_url
    if _uses_openai_compatible_chat_path(settings.elice_base_url):
        return settings.elice_base_url
    configured = ", ".join(sorted(settings.elice_model_endpoints)) or "(none)"
    raise EliceConfigError(
        "No Elice endpoint configured for requested model "
        f"{model!r}. Configure ELICE_MODEL_ENDPOINTS. Available mapped models: {configured}."
    )


def _resolve_embedding_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/embeddings"):
        return base
    if base.endswith("/chat/completions"):
        return f"{base[: -len('/chat/completions')]}/embeddings"
    if base.endswith("/v1"):
        return f"{base}/embeddings"
    if _looks_like_elice_endpoint_root(base):
        return f"{base}/v1/embeddings"
    return base


def _uses_openai_compatible_chat_path(base_url: str) -> bool:
    base = base_url.rstrip("/")
    return (
        base.endswith("/v1")
        or base.endswith("/chat/completions")
        or _looks_like_elice_endpoint_root(base)
    )


def _looks_like_elice_endpoint_root(base_url: str) -> bool:
    parsed = urlparse(base_url)
    if parsed.netloc != "mlapi.run":
        return False
    path_parts = [part for part in parsed.path.split("/") if part]
    return len(path_parts) == 1


def _extract_chat_text(data: dict[str, Any]) -> str:
    if "choices" in data and data["choices"]:
        first_choice = data["choices"][0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict) and item.get("text"):
                            parts.append(str(item["text"]))
                    text = "\n".join(part.strip() for part in parts if part.strip())
                    if text:
                        return text
                refusal = message.get("refusal")
                if isinstance(refusal, str) and refusal.strip():
                    return refusal.strip()
            if first_choice.get("text"):
                return str(first_choice["text"]).strip()

    for key in ("output_text", "response", "result", "text", "content"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise EliceConfigError("Elice chat response did not contain a supported text field.")


def _extract_chat_delta(data: dict[str, Any]) -> str:
    if "choices" in data and data["choices"]:
        first_choice = data["choices"][0]
        if isinstance(first_choice, dict):
            delta = first_choice.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict) and item.get("text"):
                            parts.append(str(item["text"]))
                    return "".join(parts)
                refusal = delta.get("refusal")
                if isinstance(refusal, str):
                    return refusal
            text = first_choice.get("text")
            if isinstance(text, str):
                return text
    for key in ("output_text", "response", "result", "text", "content"):
        value = data.get(key)
        if isinstance(value, str):
            return value
    return ""


def _post_chat_with_compat(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict,
) -> httpx.Response:
    response = client.post(url, headers=headers, json=payload)
    for _ in range(3):
        if response.status_code != 400:
            return response
        text = response.text
        changed = False
        if "max_completion_tokens" in text and "max_tokens" in payload:
            max_tokens = payload.pop("max_tokens")
            payload["max_completion_tokens"] = max_tokens
            changed = True
        if "temperature" in text and "temperature" in payload:
            payload.pop("temperature", None)
            changed = True
        if not changed:
            return response
        response = client.post(url, headers=headers, json=payload)
    return response
