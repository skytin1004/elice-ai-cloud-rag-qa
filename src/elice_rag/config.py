from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env_file(path: str | Path | None, *, override: bool = False) -> None:
    """Load a simple KEY=VALUE env file without adding a dotenv dependency."""
    if not path:
        return
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value


def load_runtime_env() -> None:
    load_env_file(PROJECT_ROOT / ".env")
    load_env_file(os.getenv("ENV_FILE"))


def parse_mapping(value: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in value.split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, mapped_value = item.split("=", 1)
        key = key.strip()
        mapped_value = mapped_value.strip()
        if key and mapped_value:
            mapping[key] = mapped_value.rstrip("/")
    return mapping


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "mock"
    embedding_provider: str = "local"
    rag_top_k: int = 5
    rag_min_score: float = 0.08
    rag_max_context_chars: int = 12000
    sources_path: Path = PROJECT_ROOT / "data" / "sources.json"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    chunks_path: Path = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
    index_path: Path = PROJECT_ROOT / "data" / "index" / "vector_index.json"

    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = ""
    azure_openai_chat_deployment_name: str = ""
    azure_openai_model_name: str = ""
    azure_openai_embedding_deployment_name: str = ""

    elice_api_key: str = ""
    elice_base_url: str = ""
    elice_chat_model: str = "openai/gpt-5-mini"
    elice_model_endpoints: dict[str, str] = field(default_factory=dict)
    elice_embedding_base_url: str = ""
    elice_embedding_model: str = "openai/text-embedding-3-small"


def get_settings() -> Settings:
    load_runtime_env()
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "mock").lower(),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local").lower(),
        rag_top_k=int(os.getenv("RAG_TOP_K", "5")),
        rag_min_score=float(os.getenv("RAG_MIN_SCORE", "0.08")),
        rag_max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000")),
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/"),
        azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", ""),
        azure_openai_chat_deployment_name=os.getenv(
            "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", ""
        ),
        azure_openai_model_name=os.getenv("AZURE_OPENAI_MODEL_NAME", ""),
        azure_openai_embedding_deployment_name=os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", ""
        ),
        elice_api_key=os.getenv("ELICE_API_KEY", ""),
        elice_base_url=os.getenv("ELICE_BASE_URL", "").rstrip("/"),
        elice_chat_model=os.getenv("ELICE_CHAT_MODEL", "openai/gpt-5-mini"),
        elice_model_endpoints=parse_mapping(os.getenv("ELICE_MODEL_ENDPOINTS", "")),
        elice_embedding_base_url=os.getenv("ELICE_EMBEDDING_BASE_URL", "").rstrip("/"),
        elice_embedding_model=os.getenv(
            "ELICE_EMBEDDING_MODEL", "openai/text-embedding-3-small"
        ),
    )
