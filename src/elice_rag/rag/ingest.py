from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

from elice_rag.config import Settings, get_settings
from elice_rag.providers import create_embedding_client

from .chunk import fixed_size_chunks, heading_aware_chunks
from .clean import clean_document
from .index import JsonVectorIndex, load_chunks, write_chunks
from .schemas import RawDocument, SourceDocument


def load_sources(path: str | Path) -> list[SourceDocument]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [SourceDocument.model_validate(item) for item in data]


def download_sources(settings: Settings) -> list[RawDocument]:
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    sources = load_sources(settings.sources_path)
    raw_documents: list[RawDocument] = []
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        for source in sources:
            response = client.get(source.url)
            response.raise_for_status()
            raw = RawDocument(
                id=source.id,
                title=source.title,
                url=str(response.url),
                fetched_at=datetime.now(timezone.utc).isoformat(),
                html=response.text,
            )
            output_path = settings.raw_dir / f"{source.id}.json"
            output_path.write_text(
                raw.model_dump_json(indent=2), encoding="utf-8"
            )
            raw_documents.append(raw)
    return raw_documents


def process_raw_documents(settings: Settings, *, strategy: str = "heading") -> int:
    chunks = []
    for raw_path in sorted(settings.raw_dir.glob("*.json")):
        raw = RawDocument.model_validate_json(raw_path.read_text(encoding="utf-8"))
        clean = clean_document(raw)
        if strategy == "fixed":
            chunks.extend(fixed_size_chunks(clean))
        else:
            chunks.extend(heading_aware_chunks(clean))
    write_chunks(chunks, settings.chunks_path)
    return len(chunks)


def build_index(settings: Settings) -> int:
    chunks = load_chunks(settings.chunks_path)
    embedding_client = create_embedding_client(settings)
    JsonVectorIndex.build(chunks, embedding_client, settings.index_path)
    return len(chunks)


def run_all(settings: Settings, *, strategy: str = "heading") -> None:
    download_sources(settings)
    process_raw_documents(settings, strategy=strategy)
    build_index(settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG ingest pipeline")
    parser.add_argument("command", choices=["download", "process", "index", "all"])
    parser.add_argument("--strategy", choices=["heading", "fixed"], default="heading")
    args = parser.parse_args()

    settings = get_settings()
    if args.command == "download":
        docs = download_sources(settings)
        print(f"downloaded={len(docs)}")
    elif args.command == "process":
        count = process_raw_documents(settings, strategy=args.strategy)
        print(f"chunks={count}")
    elif args.command == "index":
        count = build_index(settings)
        print(f"indexed_chunks={count}")
    else:
        run_all(settings, strategy=args.strategy)
        print("ingest_complete=true")


if __name__ == "__main__":
    main()

