"""End-to-end CourseRAG pipeline orchestrating ingestion, hybrid retrieval, and synthesis."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from src.chunking import chunk_pages
from src.config import DEFAULT_PDF_PATH, PARSE_CACHE_DIR, EMBED_CACHE_DIR, RAGSettings
from src.generator import contextualize_query, generate_answer
from src.ingestion import load_document_bytes, load_document_file
from src.retriever import HybridRetriever, RetrievedChunk


@dataclass
class DocumentSummary:
    """Metadata summary for an indexed document."""

    source_name: str
    page_count: int
    chunk_count: int
    file_hash: str
    is_builtin: bool = False


@dataclass
class RAGResponse:
    """Complete result returned for a single user query."""

    answer: str
    rewritten_query: str
    retrieved_chunks: list[RetrievedChunk]
    provider_used: str
    model_used: str
    retrieval_ms: float
    total_ms: float


class CourseRAGPipeline:
    """High-level interface managing knowledge base state and RAG execution."""

    def __init__(
        self,
        settings: RAGSettings | None = None,
        load_default_handbook: bool = True,
        parse_cache_dir: Path = PARSE_CACHE_DIR,
        embed_cache_dir: Path = EMBED_CACHE_DIR,
    ) -> None:
        self.settings = settings or RAGSettings()
        self.parse_cache_dir = parse_cache_dir
        self.retriever = HybridRetriever(self.settings, embed_cache_dir=embed_cache_dir)
        self._documents: dict[str, DocumentSummary] = {}

        if load_default_handbook and DEFAULT_PDF_PATH.is_file():
            self.index_file(DEFAULT_PDF_PATH, is_builtin=True)

    def index_file(self, path: Path, is_builtin: bool = False) -> DocumentSummary:
        """Parse, chunk, and index a file from disk."""
        pages = load_document_file(path, cache_dir=self.parse_cache_dir)
        return self._index_pages(pages, is_builtin=is_builtin)

    def index_bytes(self, filename: str, raw_bytes: bytes, is_builtin: bool = False) -> DocumentSummary:
        """Parse, chunk, and index raw uploaded file bytes."""
        pages = load_document_bytes(filename, raw_bytes, cache_dir=self.parse_cache_dir)
        return self._index_pages(pages, is_builtin=is_builtin)

    def _index_pages(self, pages, is_builtin: bool = False) -> DocumentSummary:
        source_name = pages[0].source_name
        file_hash = pages[0].file_hash

        # Replace prior version if re-uploading a document with the same name
        if source_name in self._documents:
            if self._documents[source_name].file_hash == file_hash:
                return self._documents[source_name]
            self.retriever.remove_source(source_name)

        chunks = chunk_pages(
            pages,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.retriever.add_chunks(chunks)

        summary = DocumentSummary(
            source_name=source_name,
            page_count=len(pages),
            chunk_count=len(chunks),
            file_hash=file_hash,
            is_builtin=is_builtin,
        )
        self._documents[source_name] = summary
        return summary

    def remove_document(self, source_name: str) -> bool:
        if source_name not in self._documents:
            return False
        self.retriever.remove_source(source_name)
        del self._documents[source_name]
        return True

    def list_documents(self) -> list[DocumentSummary]:
        return list(self._documents.values())

    def stats(self) -> dict[str, int]:
        docs = self.list_documents()
        return {
            "documents": len(docs),
            "pages": sum(d.page_count for d in docs),
            "chunks": self.retriever.total_chunks,
        }

    def ask(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        retrieval_mode: str | None = None,
        top_k: int | None = None,
        source_filter: list[str] | None = None,
    ) -> RAGResponse:
        """Run query contextualization, hybrid retrieval, and grounded generation."""
        t_start = time.perf_counter()

        rewritten = contextualize_query(question, chat_history)
        t_ret_start = time.perf_counter()
        retrieved = self.retriever.search(
            query=rewritten,
            mode=retrieval_mode or self.settings.retrieval_mode,
            top_k=top_k or self.settings.top_k,
            source_filter=source_filter,
        )
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000.0

        answer, provider_used, model_used = generate_answer(
            question=question,
            retrieved=retrieved,
            settings=self.settings,
            chat_history=chat_history,
        )
        total_ms = (time.perf_counter() - t_start) * 1000.0

        return RAGResponse(
            answer=answer,
            rewritten_query=rewritten,
            retrieved_chunks=retrieved,
            provider_used=provider_used,
            model_used=model_used,
            retrieval_ms=round(retrieval_ms, 1),
            total_ms=round(total_ms, 1),
        )
