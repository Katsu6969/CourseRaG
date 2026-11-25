"""Automated test suite for CourseRAG ingestion, chunking, hybrid retrieval, and synthesis."""

from __future__ import annotations

import io
from pathlib import Path

from docx import Document as DocxDocument
import pytest

from src.chunking import chunk_pages
from src.config import DEFAULT_PDF_PATH, RAGSettings
from src.generator import contextualize_query
from src.ingestion import load_document_bytes, load_document_file
from src.pipeline import CourseRAGPipeline


@pytest.fixture(scope="module")
def shared_pipeline(tmp_path_factory) -> CourseRAGPipeline:
    cache_root = tmp_path_factory.mktemp("rag_cache")
    settings = RAGSettings(llm_provider="local", top_k=4)
    return CourseRAGPipeline(
        settings=settings,
        load_default_handbook=True,
        parse_cache_dir=cache_root / "parsed",
        embed_cache_dir=cache_root / "embeds",
    )


def test_default_pdf_ingestion_and_caching(tmp_path: Path):
    assert DEFAULT_PDF_PATH.is_file(), "Built-in handbook PDF must exist"
    cache_dir = tmp_path / "parse_cache"

    pages_first = load_document_file(DEFAULT_PDF_PATH, cache_dir=cache_dir)
    assert len(pages_first) >= 10
    assert pages_first[0].page_number == 1
    assert any("Timsort" in p.text for p in pages_first)

    # Verify SHA-256 JSON cache file was written and reloads identically
    cache_files = list(cache_dir.glob("*.json"))
    assert len(cache_files) == 1
    pages_cached = load_document_file(DEFAULT_PDF_PATH, cache_dir=cache_dir)
    assert len(pages_cached) == len(pages_first)
    assert pages_cached[3].text == pages_first[3].text


def test_docx_and_markdown_ingestion(tmp_path: Path):
    md_bytes = (
        b"# Amortized Hashing Notes\n\n"
        b"Cuckoo hashing maintains two hash tables and achieves O(1) worst-case lookup.\n\n"
        b"Insertions relocate displaced keys along alternating paths."
    )
    md_pages = load_document_bytes("hashing_notes.md", md_bytes, cache_dir=tmp_path)
    assert len(md_pages) == 1
    assert "Cuckoo hashing" in md_pages[0].text

    buf = io.BytesIO()
    doc = DocxDocument()
    doc.add_heading("Red-Black Tree Properties", level=1)
    doc.add_paragraph("Every path from a node to its descendant NIL leaves has the same black-height.")
    doc.save(buf)

    docx_pages = load_document_bytes("rbtree.docx", buf.getvalue(), cache_dir=tmp_path)
    assert len(docx_pages) == 1
    assert "black-height" in docx_pages[0].text


def test_section_aware_chunking(tmp_path: Path):
    pages = load_document_file(DEFAULT_PDF_PATH, cache_dir=tmp_path)
    chunks = chunk_pages(pages, chunk_size=850, chunk_overlap=140)
    assert len(chunks) >= 15
    assert all(ch.char_count > 0 for ch in chunks)
    # Check that section headings were extracted onto chunks
    sections = {ch.section_title for ch in chunks}
    assert any("Timsort" in s or "Garbage Collection" in s or "Dijkstra" in s for s in sections)


def test_hybrid_vs_dense_vs_bm25_retrieval(shared_pipeline: CourseRAGPipeline):
    # Conceptual query: memory leaks from self-referencing nodes
    res_hybrid = shared_pipeline.retriever.search(
        "How are unreachable circular references freed in Python memory?",
        mode="hybrid",
        top_k=3,
    )
    assert len(res_hybrid) == 3
    combined_text = " ".join(r.chunk.text for r in res_hybrid).lower()
    assert "generational" in combined_text or "cyclic gc" in combined_text

    # Exact identifier query: bisect_left LIS
    res_bm25 = shared_pipeline.retriever.search(
        "bisect_left length_of_lis tails",
        mode="bm25",
        top_k=3,
    )
    assert len(res_bm25) >= 1
    assert "length_of_lis" in res_bm25[0].chunk.text

    # Dense-only search
    res_dense = shared_pipeline.retriever.search(
        "shortest path in weighted graph without negative edges",
        mode="dense",
        top_k=3,
    )
    assert any("Dijkstra" in r.chunk.text for r in res_dense)


def test_followup_query_contextualization():
    history = [
        {"role": "user", "content": "Explain how Timsort works in Python"},
        {"role": "assistant", "content": "Timsort combines Insertion Sort and Merge Sort..."},
    ]
    expanded = contextualize_query("What is its best-case time complexity?", history)
    assert "timsort" in expanded.lower()


def test_dynamic_upload_and_removal(shared_pipeline: CourseRAGPipeline):
    initial_count = shared_pipeline.stats()["documents"]
    notes = (
        b"# Skip List Study Sheet\n\n"
        b"A skip list is a probabilistic multi-level linked structure invented by William Pugh "
        b"that provides O(log n) average search, insertion, and deletion without tree rotations."
    )
    summary = shared_pipeline.index_bytes("skiplist_notes.md", notes)
    assert summary.source_name == "skiplist_notes.md"
    assert shared_pipeline.stats()["documents"] == initial_count + 1

    ans = shared_pipeline.ask("Who invented skip lists and what is their search complexity?")
    assert "William Pugh" in ans.answer
    assert "skiplist_notes.md" in ans.answer

    removed = shared_pipeline.remove_document("skiplist_notes.md")
    assert removed is True
    assert shared_pipeline.stats()["documents"] == initial_count


def test_local_synthesis_citations_and_offtopic_guard(shared_pipeline: CourseRAGPipeline):
    in_domain = shared_pipeline.ask("How does heapq.heapify achieve O(n) complexity?")
    assert "Python_DSA_Reference_Guide.pdf" in in_domain.answer
    assert "heapify" in in_domain.answer

    off_topic = shared_pipeline.ask("What temperature should I bake chocolate chip cookies?")
    assert "could not find relevant coverage" in off_topic.answer.lower()
