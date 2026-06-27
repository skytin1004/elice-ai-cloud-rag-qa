from __future__ import annotations

from .schemas import RetrievedChunk


SYSTEM_PROMPT = """You are a citation-based RAG QA assistant.
Answer in Korean.
Use only the provided context.
If the context does not contain enough evidence, say that the provided documents do not contain enough information.
Do not invent facts, URLs, prices, or procedures."""


def build_messages(question: str, chunks: list[RetrievedChunk], *, max_context_chars: int) -> list[dict[str, str]]:
    context_parts: list[str] = []
    used = 0
    for idx, chunk in enumerate(chunks, 1):
        block = (
            f"[{idx}] title={chunk.title}\n"
            f"url={chunk.source_url}\n"
            f"heading={chunk.heading}\n"
            f"chunk_id={chunk.chunk_id}\n"
            f"{chunk.text}\n"
        )
        if used + len(block) > max_context_chars:
            break
        context_parts.append(block)
        used += len(block)
    context = "\n---\n".join(context_parts)
    user_prompt = f"""<context>
{context}
</context>

Question: {question}

Write a concise grounded answer. Mention when the evidence is insufficient."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

