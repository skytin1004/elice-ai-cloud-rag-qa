from __future__ import annotations

import hashlib
import re

from .schemas import Chunk, CleanDocument


WORD_RE = re.compile(r"[\w가-힣]+", re.UNICODE)


def token_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _stable_chunk_id(source_id: str, index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{source_id}-{index:03d}-{digest}"


def _split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?。])\s+|\n+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def heading_aware_chunks(
    document: CleanDocument,
    *,
    target_tokens: int = 750,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    sections: list[tuple[str, str]] = []
    current_heading = document.title
    current_lines: list[str] = []
    for raw_line in document.text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("## "):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines)))
            current_heading = line.removeprefix("## ").strip() or document.title
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines)))
    if not sections and document.text:
        sections.append((document.title, document.text))

    chunks: list[Chunk] = []
    for heading, section_text in sections:
        sentences = _split_sentences(section_text)
        buffer: list[str] = []
        for sentence in sentences:
            candidate = "\n".join(buffer + [sentence])
            if buffer and token_count(candidate) > target_tokens:
                chunks.append(
                    _make_chunk(document, heading, "\n".join(buffer), len(chunks))
                )
                buffer = _overlap_tail(buffer, overlap_tokens)
            buffer.append(sentence)
        if buffer:
            chunks.append(_make_chunk(document, heading, "\n".join(buffer), len(chunks)))
    return chunks


def fixed_size_chunks(
    document: CleanDocument,
    *,
    target_tokens: int = 750,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    words = WORD_RE.findall(document.text)
    if not words:
        return []
    chunks: list[Chunk] = []
    step = max(1, target_tokens - overlap_tokens)
    for start in range(0, len(words), step):
        window = words[start : start + target_tokens]
        if not window:
            break
        text = " ".join(window)
        chunks.append(_make_chunk(document, document.title, text, len(chunks)))
        if start + target_tokens >= len(words):
            break
    return chunks


def _overlap_tail(sentences: list[str], overlap_tokens: int) -> list[str]:
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        total += token_count(sentence)
        tail.insert(0, sentence)
        if total >= overlap_tokens:
            break
    return tail


def _make_chunk(document: CleanDocument, heading: str, text: str, index: int) -> Chunk:
    return Chunk(
        chunk_id=_stable_chunk_id(document.id, index, text),
        source_id=document.id,
        source_url=document.url,
        title=document.title,
        heading=heading,
        text=text.strip(),
        chunk_index=index,
    )

