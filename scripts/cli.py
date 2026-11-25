"""Command-line interface for querying CourseRAG from a terminal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import RAGSettings
from src.pipeline import CourseRAGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CourseRAG CLI — Hybrid Dense + BM25 Technical Study Assistant"
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default=None,
        help="Run a single question and exit (omit for interactive REPL).",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["hybrid", "dense", "bm25"],
        default="hybrid",
        help="Retrieval mode (default: hybrid).",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5).",
    )
    parser.add_argument(
        "-f",
        "--file",
        action="append",
        default=[],
        help="Optional path to an extra PDF/DOCX/MD/TXT file to index.",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print retrieved chunk scores and provenance after each answer.",
    )
    return parser.parse_args()


def _print_response(resp, show_sources: bool) -> None:
    print("\n" + "=" * 72)
    print(resp.answer)
    print("-" * 72)
    print(
        f"[Mode: {resp.provider_used} ({resp.model_used}) | "
        f"Retrieval: {resp.retrieval_ms} ms | Total: {resp.total_ms} ms]"
    )
    if show_sources and resp.retrieved_chunks:
        print("\nRetrieved Chunks:")
        for idx, item in enumerate(resp.retrieved_chunks, start=1):
            ch = item.chunk
            print(
                f"  {idx}. {ch.source_name} (p. {ch.page_number}, {ch.section_title}) "
                f"-> score={item.score} [dense={item.dense_score}, bm25={item.bm25_score}]"
            )
    print("=" * 72)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    settings = RAGSettings(retrieval_mode=args.mode, top_k=args.top_k)
    pipeline = CourseRAGPipeline(settings=settings, load_default_handbook=True)

    for extra_path in args.file:
        doc_summary = pipeline.index_file(Path(extra_path))
        print(
            f"Indexed '{doc_summary.source_name}' "
            f"({doc_summary.page_count} pages, {doc_summary.chunk_count} chunks)."
        )

    if args.query:
        result = pipeline.ask(args.query, retrieval_mode=args.mode, top_k=args.top_k)
        _print_response(result, show_sources=args.show_sources)
        return 0

    stats = pipeline.stats()
    print(
        f"CourseRAG ready — {stats['documents']} document(s), "
        f"{stats['pages']} pages, {stats['chunks']} chunks indexed."
    )
    print("Type a question to search, or 'exit' to quit.\n")

    history: list[dict[str, str]] = []
    while True:
        try:
            q = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if q.lower() in {"exit", "quit", "q"}:
            break
        if not q:
            continue

        resp = pipeline.ask(
            question=q,
            chat_history=history,
            retrieval_mode=args.mode,
            top_k=args.top_k,
        )
        _print_response(resp, show_sources=args.show_sources)
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": resp.answer})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
