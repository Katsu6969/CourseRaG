"""Hybrid dense + sparse BM25 retriever with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

from src.chunking import TextChunk
from src.config import EMBED_CACHE_DIR, RAGSettings

_EMBED_MODEL_CACHE: dict[str, TextEmbedding] = {}

_QUERY_FILLER_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "how", "why", "when", "where", "who", "which", "please",
    "explain", "tell", "me", "about", "can", "you", "describe", "give",
    "does", "do", "did", "of", "in", "on", "for", "to", "with", "and", "or",
    "i", "my", "we", "our", "should", "would", "could", "will", "shall", "may",
}


def get_embedding_model(model_name: str) -> TextEmbedding:
    """Load the local ONNX FastEmbed model once per process."""
    if model_name not in _EMBED_MODEL_CACHE:
        _EMBED_MODEL_CACHE[model_name] = TextEmbedding(model_name=model_name)
    return _EMBED_MODEL_CACHE[model_name]


def code_aware_tokenize(text: str) -> list[str]:
    """Tokenize prose, code identifiers, and complexity expressions for BM25."""
    lowered = text.lower()
    raw_tokens = re.findall(r"[a-z0-9_]+(?:\([a-z0-9_\s*+]+\))?", lowered)
    tokens: list[str] = []
    for tok in raw_tokens:
        cleaned = tok.strip("_ ")
        if not cleaned:
            continue
        if cleaned in _QUERY_FILLER_WORDS:
            continue
        tokens.append(cleaned)
        # Also split snake_case identifiers so both 'lru_cache' and 'cache' match
        if "_" in cleaned:
            for part in cleaned.split("_"):
                if part and part not in _QUERY_FILLER_WORDS and len(part) > 1:
                    tokens.append(part)
    return tokens


@dataclass
class RetrievedChunk:
    """Single retrieved chunk along with component and fused relevance scores."""

    chunk: TextChunk
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    dense_rank: int | None = None
    bm25_rank: int | None = None


class HybridRetriever:
    """Manages Qdrant dense vectors + BM25 lexical index with disk-cached embeddings."""

    def __init__(self, settings: RAGSettings, embed_cache_dir: Path = EMBED_CACHE_DIR) -> None:
        self.settings = settings
        self.embed_cache_dir = embed_cache_dir
        self.embed_cache_dir.mkdir(parents=True, exist_ok=True)

        self.embedder = get_embedding_model(settings.embedding_model)
        self.client = QdrantClient(location=":memory:")
        self._reset_qdrant_collection()

        self._chunks_by_id: dict[str, TextChunk] = {}
        self._ordered_ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[list[str]] = []

    def _reset_qdrant_collection(self) -> None:
        if self.client.collection_exists(self.settings.collection_name):
            self.client.delete_collection(self.settings.collection_name)
        self.client.create_collection(
            collection_name=self.settings.collection_name,
            vectors_config=VectorParams(
                size=self.settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )

    def _embed_chunks_with_cache(self, file_hash: str, chunks: list[TextChunk]) -> np.ndarray:
        """Load cached embeddings for this file + chunk configuration or compute via FastEmbed."""
        cache_key = f"{file_hash[:20]}_{len(chunks)}_{self.settings.chunk_size}.npz"
        cache_file = self.embed_cache_dir / cache_key

        if cache_file.is_file():
            try:
                loaded = np.load(cache_file)
                ids = loaded["ids"].tolist()
                vecs = loaded["vectors"]
                expected_ids = [c.chunk_id for c in chunks]
                if ids == expected_ids and vecs.shape == (len(chunks), self.settings.embedding_dim):
                    return vecs
            except Exception:
                pass

        texts = [c.embedding_input for c in chunks]
        vectors = np.array(list(self.embedder.embed(texts)), dtype=np.float32)
        np.savez_compressed(
            cache_file,
            ids=np.array([c.chunk_id for c in chunks]),
            vectors=vectors,
        )
        return vectors

    def add_chunks(self, chunks: list[TextChunk]) -> int:
        """Index new chunks into Qdrant and rebuild the BM25 lexical corpus."""
        if not chunks:
            return 0

        # Group by file_hash so per-document vector cache works cleanly
        grouped: dict[str, list[TextChunk]] = {}
        for ch in chunks:
            if ch.chunk_id in self._chunks_by_id:
                continue
            grouped.setdefault(ch.file_hash, []).append(ch)

        added = 0
        points: list[PointStruct] = []
        for fhash, doc_chunks in grouped.items():
            vectors = self._embed_chunks_with_cache(fhash, doc_chunks)
            for ch, vec in zip(doc_chunks, vectors):
                self._chunks_by_id[ch.chunk_id] = ch
                points.append(
                    PointStruct(
                        id=ch.chunk_id,
                        vector=vec.tolist(),
                        payload={
                            "source_name": ch.source_name,
                            "page_number": ch.page_number,
                            "section_title": ch.section_title,
                            "file_hash": ch.file_hash,
                        },
                    )
                )
                added += 1

        if points:
            self.client.upsert(
                collection_name=self.settings.collection_name,
                points=points,
            )
            self._rebuild_bm25()

        return added

    def remove_source(self, source_name: str) -> int:
        """Remove all chunks belonging to a specific document source."""
        to_remove = [
            cid for cid, ch in self._chunks_by_id.items() if ch.source_name == source_name
        ]
        if not to_remove:
            return 0

        for cid in to_remove:
            del self._chunks_by_id[cid]

        self.client.delete(
            collection_name=self.settings.collection_name,
            points_selector=to_remove,
        )
        self._rebuild_bm25()
        return len(to_remove)

    def clear_all(self) -> None:
        self._chunks_by_id.clear()
        self._ordered_ids.clear()
        self._bm25 = None
        self._bm25_corpus.clear()
        self._reset_qdrant_collection()

    def _rebuild_bm25(self) -> None:
        self._ordered_ids = list(self._chunks_by_id.keys())
        if not self._ordered_ids:
            self._bm25 = None
            self._bm25_corpus = []
            return
        self._bm25_corpus = [
            code_aware_tokenize(self._chunks_by_id[cid].embedding_input)
            for cid in self._ordered_ids
        ]
        self._bm25 = BM25Okapi(self._bm25_corpus)

    @property
    def total_chunks(self) -> int:
        return len(self._chunks_by_id)

    def all_chunks(self) -> list[TextChunk]:
        return list(self._chunks_by_id.values())

    def search(
        self,
        query: str,
        mode: str | None = None,
        top_k: int | None = None,
        source_filter: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks via 'hybrid', 'dense', or 'bm25' strategy."""
        if not self._chunks_by_id or not query.strip():
            return []

        active_mode = (mode or self.settings.retrieval_mode or "hybrid").lower()
        k = top_k or self.settings.top_k
        candidate_pool = max(k * 3, 15)

        # 1. Dense semantic search via Qdrant
        dense_scores: dict[str, float] = {}
        dense_ranks: dict[str, int] = {}
        if active_mode in {"hybrid", "dense"}:
            q_vec = next(iter(self.embedder.embed([query]))).tolist()
            q_filter = None
            if source_filter:
                q_filter = Filter(
                    must=[
                        FieldCondition(
                            key="source_name",
                            match=MatchAny(any=source_filter),
                        )
                    ]
                )
            response = self.client.query_points(
                collection_name=self.settings.collection_name,
                query=q_vec,
                query_filter=q_filter,
                limit=candidate_pool,
            )
            for rank_idx, pt in enumerate(response.points, start=1):
                cid = str(pt.id)
                if cid in self._chunks_by_id:
                    dense_scores[cid] = float(pt.score)
                    dense_ranks[cid] = rank_idx

        # 2. Sparse lexical search via BM25Okapi
        bm25_scores: dict[str, float] = {}
        bm25_ranks: dict[str, int] = {}
        if active_mode in {"hybrid", "bm25"} and self._bm25 is not None:
            q_tokens = code_aware_tokenize(query)
            if q_tokens:
                raw_scores = self._bm25.get_scores(q_tokens)
                allowed_sources = set(source_filter) if source_filter else None
                scored_pairs: list[tuple[str, float]] = []
                for cid, sc in zip(self._ordered_ids, raw_scores):
                    if sc <= 0.0:
                        continue
                    if allowed_sources and self._chunks_by_id[cid].source_name not in allowed_sources:
                        continue
                    scored_pairs.append((cid, float(sc)))

                scored_pairs.sort(key=lambda item: item[1], reverse=True)
                for rank_idx, (cid, sc) in enumerate(scored_pairs[:candidate_pool], start=1):
                    bm25_scores[cid] = sc
                    bm25_ranks[cid] = rank_idx

        # 3. Score aggregation / Reciprocal Rank Fusion
        if active_mode == "dense":
            ranked_ids = sorted(dense_scores.keys(), key=lambda c: dense_scores[c], reverse=True)[:k]
            return [
                RetrievedChunk(
                    chunk=self._chunks_by_id[cid],
                    score=round(dense_scores[cid], 4),
                    dense_score=round(dense_scores[cid], 4),
                    bm25_score=round(bm25_scores.get(cid, 0.0), 4),
                    dense_rank=dense_ranks.get(cid),
                    bm25_rank=bm25_ranks.get(cid),
                )
                for cid in ranked_ids
            ]

        if active_mode == "bm25":
            ranked_ids = sorted(bm25_scores.keys(), key=lambda c: bm25_scores[c], reverse=True)[:k]
            return [
                RetrievedChunk(
                    chunk=self._chunks_by_id[cid],
                    score=round(bm25_scores[cid], 4),
                    dense_score=round(dense_scores.get(cid, 0.0), 4),
                    bm25_score=round(bm25_scores[cid], 4),
                    dense_rank=dense_ranks.get(cid),
                    bm25_rank=bm25_ranks.get(cid),
                )
                for cid in ranked_ids
            ]

        # Weighted Reciprocal Rank Fusion (RRF): score(d) = w_d / (rrf_k + r_d) + w_s / (rrf_k + r_s)
        all_candidates = set(dense_ranks.keys()) | set(bm25_ranks.keys())
        rrf_scores: dict[str, float] = {}
        rrf_const = self.settings.rrf_k
        for cid in all_candidates:
            fused = 0.0
            if cid in dense_ranks:
                fused += self.settings.dense_weight / (rrf_const + dense_ranks[cid])
            if cid in bm25_ranks:
                fused += self.settings.sparse_weight / (rrf_const + bm25_ranks[cid])
            rrf_scores[cid] = fused

        ranked_ids = sorted(
            rrf_scores.keys(),
            key=lambda c: (rrf_scores[c], dense_scores.get(c, 0.0)),
            reverse=True,
        )[:k]

        return [
            RetrievedChunk(
                chunk=self._chunks_by_id[cid],
                score=round(rrf_scores[cid], 5),
                dense_score=round(dense_scores.get(cid, 0.0), 4),
                bm25_score=round(bm25_scores.get(cid, 0.0), 4),
                dense_rank=dense_ranks.get(cid),
                bm25_rank=bm25_ranks.get(cid),
            )
            for cid in ranked_ids
        ]
