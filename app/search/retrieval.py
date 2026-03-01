import asyncio
import logging
import re
import time
from typing import List, Dict, Any, Optional, Set
from app.storage.db import DatabaseManager
from app.embeddings.service import EmbeddingService
from app.vector_store.chroma_client import ChromaClient
from app.search.context_builder import build_context
from app.search.llm_client import LLMClient
from app.search.reranker import rerank
from app.config import settings

logger = logging.getLogger(__name__)

# Characters that act as FTS5 operators and must be stripped from user input
_FTS5_OPERATOR_RE = re.compile(r'["*^]|\bAND\b|\bOR\b|\bNOT\b|\bNEAR\b', re.IGNORECASE)


def _sanitize_fts_query(query: str) -> str:
    """Sanitize user input for safe use in FTS5 MATCH.

    Strips FTS5 operators (AND, OR, NOT, NEAR, *, ^, ") then wraps each
    token in double-quotes so the query is treated as literal terms.
    """
    cleaned = _FTS5_OPERATOR_RE.sub(' ', query)
    tokens = [t.strip() for t in cleaned.split() if t.strip()]
    if not tokens:
        return '"' + query.replace('"', '') + '"'
    return ' '.join(f'"{t}"' for t in tokens)


async def _fts_search(db: DatabaseManager, query: str, k: int) -> List[Dict[str, Any]]:
    """Run FTS5 keyword search."""
    try:
        fts_match = _sanitize_fts_query(query)
        fts_sql = "SELECT rowid, chunks_text FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY rank LIMIT ?"
        rows = await db.execute_query(fts_sql, (fts_match, 2 * k))
        return [{"id": str(row[0]), "text": row[1]} for row in rows]
    except Exception as e:
        logger.warning("FTS5 Search failed (non-fatal, falling back to semantic only): %s", e)
        return []


async def _semantic_search_with_emb(
    chroma_client: ChromaClient,
    query_emb: List[float],
    k: int,
) -> List[Dict[str, Any]]:
    """Run Chroma semantic search using a pre-computed embedding."""
    raw = await chroma_client.semantic_search(query_emb, k=2 * k)
    results: List[Dict[str, Any]] = []
    if raw.get("ids") and raw["ids"][0]:
        ids = raw["ids"][0]
        dists = raw.get("distances", [[]])[0]
        for i, doc_id in enumerate(ids):
            results.append({
                "id": doc_id,
                "score": dists[i] if i < len(dists) else 0.0,
            })
    return results


def _compute_rrf_scores(
    fts_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    k: int,
) -> List[tuple]:
    """Compute Reciprocal Rank Fusion scores and return top-k (id, score) pairs.
    
    Weights and k_rrf are configurable via settings.
    """
    scores: Dict[str, float] = {}
    k_rrf = settings.rrf_k
    fts_w = settings.rrf_fts_weight
    sem_w = settings.rrf_semantic_weight
    for rank, res in enumerate(fts_results):
        scores[res["id"]] = fts_w * (1.0 / (k_rrf + rank + 1))
    for rank, res in enumerate(semantic_results):
        chunk_id = res["id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + sem_w * (1.0 / (k_rrf + rank + 1))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


async def _summary_search_with_emb(
    chroma_client: ChromaClient,
    query_emb: List[float],
    k: int,
) -> Set[str]:
    """Search the document-summary collection and return file paths of top-k docs.

    These paths are used to *boost* chunks from relevant documents during
    RRF scoring (two-stage retrieval: doc → chunk).
    """
    try:
        raw = await chroma_client.search_summaries(query_emb, k=k)
        paths: Set[str] = set()
        if raw.get("metadatas") and raw["metadatas"][0]:
            for meta in raw["metadatas"][0]:
                fp = meta.get("file_path")
                if fp:
                    paths.add(fp)
        return paths
    except Exception as e:
        logger.debug("Summary search unavailable (non-fatal): %s", e)
        return set()


async def hybrid_retrieve(
    query: str, 
    db: DatabaseManager, 
    embedding_service: EmbeddingService, 
    chroma_client: ChromaClient,
    k: int = settings.retrieval_top_k,
) -> List[Dict[str, Any]]:
    """Combines FTS5 keyword + Chroma semantic + document-summary search using RRF,
    then reranks the candidates with a cross-encoder for maximum precision."""

    # ── Compute query embedding once, reuse for semantic + summary ────────
    query_emb = await embedding_service.embed_query(query)

    # ── Run FTS5, Semantic, and Summary searches in parallel ─────────────
    fts_results, semantic_results, relevant_doc_paths = await asyncio.gather(
        _fts_search(db, query, k),
        _semantic_search_with_emb(chroma_client, query_emb, k),
        _summary_search_with_emb(chroma_client, query_emb, k),
    )

    # ── Reciprocal Rank Fusion (RRF) with summary boost ──────────────────
    sorted_ids = _compute_rrf_scores(fts_results, semantic_results, k * 2)

    # ── Fetch full metadata from DB (batched) ────────────────────────────
    if not sorted_ids:
        return []

    chunk_ids_ordered = [int(cid) for cid, _ in sorted_ids]
    score_map = {int(cid): sc for cid, sc in sorted_ids}
    placeholders = ",".join("?" for _ in chunk_ids_ordered)
    query_sql = (
        f"SELECT c.id, c.text_preview, f.path, f.folder_tag "
        f"FROM chunks c JOIN files f ON c.file_id = f.id "
        f"WHERE c.id IN ({placeholders})"
    )
    rows = await db.execute_query(query_sql, tuple(chunk_ids_ordered))
    row_map: Dict[int, Any] = {}
    for row in rows:
        row_map[row[0]] = row

    # Preserve RRF ranking order and apply summary boost
    results = []
    for cid in chunk_ids_ordered:
        row = row_map.get(cid)
        if row:
            file_path = row[2]
            rrf_score = score_map[cid] * settings.rrf_score_scale
            # Boost chunks from documents that the summary search deemed relevant
            if file_path in relevant_doc_paths:
                rrf_score *= settings.summary_boost_factor
            results.append({
                "chunk_id": cid,
                "text": row[1],
                "file_path": file_path,
                "folder_tag": row[3],
                "score": round(rrf_score, 4),
            })

    # ── Rerank with cross-encoder ────────────────────────────────────────
    if results:
        results = await rerank(query, results, top_k=k, text_key="text")

    return results

async def full_rag(
    query: str,
    db: DatabaseManager,
    embedding_service: EmbeddingService,
    chroma_client: ChromaClient,
    llm_client: LLMClient,
    k: int = settings.retrieval_top_k,
    file_type: Optional[str] = None,
    folder_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieves context and generates an LLM answer.
    
    Supports optional file_type and folder_tag filters.
    Tracks latency for each RAG pipeline stage.
    """
    t_start = time.perf_counter()
    
    # Step 1: Hybrid Retrieval
    retrieved = await hybrid_retrieve(
        query=query,
        db=db,
        embedding_service=embedding_service,
        chroma_client=chroma_client,
        k=k,
    )
    t_retrieval = time.perf_counter()
    
    # Apply optional post-retrieval filters
    if file_type or folder_tag:
        filtered = []
        for r in retrieved:
            fp = r.get("file_path", "")
            ft = r.get("folder_tag", "")
            if file_type and not fp.lower().endswith(file_type):
                continue
            if folder_tag and ft != folder_tag:
                continue
            filtered.append(r)
        retrieved = filtered
    
    if not retrieved:
        return {
            "answer": "I couldn't find any relevant documents to answer your question.",
            "sources": [],
            "retrieved_count": 0,
            "latency_ms": round((time.perf_counter() - t_start) * 1000, 1),
        }
        
    # Step 2: Build Context
    context = build_context(retrieved)
    t_context = time.perf_counter()
    
    # Step 3: LLM Generation
    answer = await llm_client.generate_answer(query, context)
    t_llm = time.perf_counter()
    
    total_ms = round((t_llm - t_start) * 1000, 1)
    logger.info(
        "RAG pipeline: retrieval=%.0fms context=%.0fms llm=%.0fms total=%.0fms",
        (t_retrieval - t_start) * 1000,
        (t_context - t_retrieval) * 1000,
        (t_llm - t_context) * 1000,
        total_ms,
    )
    
    # Step 4: Parse answer and return
    return {
        "answer": answer,
        "sources": retrieved,
        "retrieved_count": len(retrieved),
        "latency_ms": total_ms,
    }
