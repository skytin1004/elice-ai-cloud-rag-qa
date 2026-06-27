from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import re

from elice_rag.config import Settings, get_settings
from elice_rag.providers import create_chat_client, create_embedding_client

from .citations import citations_from_chunks
from .index import JsonVectorIndex
from .prompts import build_messages
from .schemas import QueryResponse, RetrievedChunk


INSUFFICIENT_ANSWER = "제공된 문서에서 해당 내용을 확인할 수 없습니다."
TOKEN_RE = re.compile(r"[\w가-힣]+", re.UNICODE)
STOPWORDS = {
    "어디",
    "어떤",
    "무엇",
    "어떻게",
    "하나요",
    "있나요",
    "수",
    "있",
    "되나요",
    "설명",
    "주세요",
    "위해",
    "해야",
    "인가요",
    "나요",
}


@dataclass
class RAGPipeline:
    settings: Settings

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RAGPipeline":
        return cls(settings or get_settings())

    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        model: str | None = None,
    ) -> QueryResponse:
        embedding_client = create_embedding_client(self.settings)
        chat_client = create_chat_client(self.settings)
        selected_model = model or chat_client.model_name
        index = JsonVectorIndex.load(self.settings.index_path)
        retrieved = index.search(
            question,
            embedding_client,
            top_k=top_k or self.settings.rag_top_k,
        )
        usable = self._usable_chunks(question, retrieved)
        if not usable:
            return QueryResponse(
                status="insufficient_context",
                answer=INSUFFICIENT_ANSWER,
                citations=[],
                confidence="low",
                retrieved_context=retrieved,
                model=selected_model,
            )

        messages = build_messages(
            question,
            usable,
            max_context_chars=self.settings.rag_max_context_chars,
        )
        answer = chat_client.generate(
            messages,
            temperature=0.0,
            model=selected_model,
        ).strip()
        if self._looks_insufficient(answer):
            return QueryResponse(
                status="insufficient_context",
                answer=INSUFFICIENT_ANSWER,
                citations=[],
                confidence="low",
                retrieved_context=retrieved,
                model=selected_model,
            )
        citations = citations_from_chunks(usable)
        return QueryResponse(
            status="answered",
            answer=answer,
            citations=citations,
            confidence=self._confidence(usable),
            retrieved_context=retrieved,
            model=selected_model,
        )

    def stream_query_events(
        self,
        question: str,
        *,
        top_k: int | None = None,
        model: str | None = None,
    ) -> Iterator[dict]:
        embedding_client = create_embedding_client(self.settings)
        chat_client = create_chat_client(self.settings)
        selected_model = model or chat_client.model_name
        index = JsonVectorIndex.load(self.settings.index_path)
        retrieved = index.search(
            question,
            embedding_client,
            top_k=top_k or self.settings.rag_top_k,
        )
        usable = self._usable_chunks(question, retrieved)
        if not usable:
            response = QueryResponse(
                status="insufficient_context",
                answer=INSUFFICIENT_ANSWER,
                citations=[],
                confidence="low",
                retrieved_context=retrieved,
                model=selected_model,
            )
            yield {"event": "done", "data": response.model_dump()}
            return

        citations = citations_from_chunks(usable)
        confidence = self._confidence(usable)
        yield {
            "event": "metadata",
            "data": {
                "status": "answering",
                "citations": [citation.model_dump() for citation in citations],
                "confidence": confidence,
                "model": selected_model,
            },
        }
        messages = build_messages(
            question,
            usable,
            max_context_chars=self.settings.rag_max_context_chars,
        )
        parts: list[str] = []
        for token in chat_client.stream(messages, temperature=0.0, model=selected_model):
            if not token:
                continue
            parts.append(token)
            yield {"event": "token", "data": {"text": token}}

        answer = "".join(parts).strip()
        if self._looks_insufficient(answer):
            response = QueryResponse(
                status="insufficient_context",
                answer=INSUFFICIENT_ANSWER,
                citations=[],
                confidence="low",
                retrieved_context=retrieved,
                model=selected_model,
            )
        else:
            response = QueryResponse(
                status="answered",
                answer=answer,
                citations=citations,
                confidence=confidence,
                retrieved_context=retrieved,
                model=selected_model,
            )
        yield {"event": "done", "data": response.model_dump()}

    def _usable_chunks(
        self, question: str, retrieved: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        return [
            chunk
            for chunk in retrieved
            if chunk.score >= self.settings.rag_min_score
            and self._has_keyword_overlap(question, chunk)
        ]

    @staticmethod
    def _has_keyword_overlap(question: str, chunk: RetrievedChunk) -> bool:
        question_terms = {
            term
            for term in TOKEN_RE.findall(question.lower())
            if len(term) >= 2 and term not in STOPWORDS
        }
        if not question_terms:
            return True
        haystack = f"{chunk.title}\n{chunk.heading}\n{chunk.text}".lower()
        hits = {term for term in question_terms if term in haystack}
        return len(hits) >= 1 and (len(hits) / len(question_terms) >= 0.2)

    @staticmethod
    def _looks_insufficient(answer: str) -> bool:
        stripped = answer.strip()
        lowered = stripped.lower()
        strong_prefixes = [
            "제공된 문서에서 해당 내용을 확인할 수 없습니다",
            "제공된 문서만으로는 답변할 수 없습니다",
            "문서에서 해당 내용을 확인할 수 없습니다",
            "정보가 부족합니다",
            "insufficient context",
            "i don't know",
            "i cannot answer",
        ]
        if any(lowered.startswith(prefix.lower()) for prefix in strong_prefixes):
            return True

        short_answer_markers = [
            "확인할 수 없습니다",
            "정보가 없습니다",
            "insufficient",
            "not contain",
            "cannot determine",
        ]
        return len(stripped) < 60 and any(
            marker in lowered for marker in short_answer_markers
        )

    @staticmethod
    def _confidence(chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "low"
        best = chunks[0].score
        if best >= 0.35:
            return "high"
        if best >= 0.16:
            return "medium"
        return "low"
