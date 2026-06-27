from __future__ import annotations

from elice_rag.config import Settings, get_settings

from .azure_openai import AzureOpenAIChatClient, AzureOpenAIEmbeddingClient
from .base import ChatClient, EmbeddingClient
from .elice import EliceChatClient, EliceEmbeddingClient
from .local_embeddings import HashingEmbeddingClient
from .mock import MockChatClient, MockEmbeddingClient


def create_chat_client(settings: Settings | None = None) -> ChatClient:
    settings = settings or get_settings()
    provider = settings.llm_provider
    if provider == "azure":
        return AzureOpenAIChatClient(settings)
    if provider == "elice":
        return EliceChatClient(settings)
    if provider == "mock":
        return MockChatClient()
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def create_embedding_client(settings: Settings | None = None) -> EmbeddingClient:
    settings = settings or get_settings()
    provider = settings.embedding_provider
    if provider == "azure":
        return AzureOpenAIEmbeddingClient(settings)
    if provider == "elice":
        return EliceEmbeddingClient(settings)
    if provider == "mock":
        return MockEmbeddingClient()
    if provider == "local":
        return HashingEmbeddingClient()
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")

