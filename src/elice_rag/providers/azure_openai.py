from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import httpx

from elice_rag.config import Settings


class AzureOpenAIConfigError(RuntimeError):
    pass


@dataclass
class AzureOpenAIChatClient:
    settings: Settings
    provider_name: str = "azure"

    @property
    def model_name(self) -> str:
        return (
            self.settings.azure_openai_model_name
            or self.settings.azure_openai_chat_deployment_name
        )

    def _validate(self) -> None:
        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_API_KEY": self.settings.azure_openai_api_key,
                "AZURE_OPENAI_ENDPOINT": self.settings.azure_openai_endpoint,
                "AZURE_OPENAI_API_VERSION": self.settings.azure_openai_api_version,
                "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": self.settings.azure_openai_chat_deployment_name,
            }.items()
            if not value
        ]
        if missing:
            raise AzureOpenAIConfigError(
                "Missing Azure OpenAI settings: " + ", ".join(missing)
            )

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str:
        self._validate()
        url = (
            f"{self.settings.azure_openai_endpoint}/openai/deployments/"
            f"{self.settings.azure_openai_chat_deployment_name}/chat/completions"
        )
        params = {"api-version": self.settings.azure_openai_api_version}
        headers = {
            "api-key": self.settings.azure_openai_api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 800,
        }
        with httpx.Client(timeout=60) as client:
            response = _post_chat_with_compat(
                client,
                url,
                headers=headers,
                payload=payload,
                params=params,
            )
            response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> Iterator[str]:
        yield self.generate(messages, temperature=temperature, model=model)


@dataclass
class AzureOpenAIEmbeddingClient:
    settings: Settings
    provider_name: str = "azure"

    @property
    def model_name(self) -> str:
        return self.settings.azure_openai_embedding_deployment_name

    def _validate(self) -> None:
        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_API_KEY": self.settings.azure_openai_api_key,
                "AZURE_OPENAI_ENDPOINT": self.settings.azure_openai_endpoint,
                "AZURE_OPENAI_API_VERSION": self.settings.azure_openai_api_version,
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": self.settings.azure_openai_embedding_deployment_name,
            }.items()
            if not value
        ]
        if missing:
            raise AzureOpenAIConfigError(
                "Missing Azure OpenAI embedding settings: " + ", ".join(missing)
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._validate()
        url = (
            f"{self.settings.azure_openai_endpoint}/openai/deployments/"
            f"{self.settings.azure_openai_embedding_deployment_name}/embeddings"
        )
        params = {"api-version": self.settings.azure_openai_api_version}
        headers = {
            "api-key": self.settings.azure_openai_api_key,
            "Content-Type": "application/json",
        }
        payload = {"input": texts}
        with httpx.Client(timeout=60) as client:
            response = client.post(url, params=params, headers=headers, json=payload)
            response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]


def _post_chat_with_compat(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict,
    params: dict[str, str],
) -> httpx.Response:
    response = client.post(url, params=params, headers=headers, json=payload)
    for _ in range(3):
        if response.status_code != 400:
            return response
        text = response.text
        changed = False
        if "max_completion_tokens" in text and "max_tokens" in payload:
            payload.pop("max_tokens", None)
            payload["max_completion_tokens"] = 800
            changed = True
        if "temperature" in text and "temperature" in payload:
            payload.pop("temperature", None)
            changed = True
        if not changed:
            return response
        response = client.post(url, params=params, headers=headers, json=payload)
    return response
