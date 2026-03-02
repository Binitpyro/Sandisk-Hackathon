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

_INVENTORY_RE = re.compile(
    r'\b(?:what|which|list|show|how many|count|do i have|files? do i|'
    r'files? i have|my files|all files|all my)\b',
    re.IGNORECASE,
)

_PROJECT_RE = re.compile(
    r'\b(?:project|projects|describe|overview|about|summary|folder|'
    r'tell me about|what is|what are|unreal|unity|godot|react|node|'
    r'python|rust|java|c\+\+|go|flutter|workspace)\b',
    re.IGNORECASE,
)

_UNREAL_RE = re.compile(r'\b(?:unreal|uasset|umap|uproject|ue5|ue4|engine)\b', re.IGNORECASE)

def _is_inventory_query(query: str) -> bool:
    """Heuristic: does the user want file-level stats rather than content?"""
    return bool(_INVENTORY_RE.search(query))

def _is_project_query(query: str) -> bool:
    """Heuristic: does the user want project-level information?"""
    return bool(_PROJECT_RE.search(query))


def _is_unreal_query(query: str) -> bool:
    """Heuristic: does the user explicitly ask about Unreal Engine data?"""
    return bool(_UNREAL_RE.search(query))


def _append_unreal_fact_lines(lines: List[str], unreal_facts: List[Dict[str, Any]]) -> None:
    lines.append("Unreal project summary:")
    for item in unreal_facts[:5]:
        lines.append(
            f"- {item['project_name']} ({item['folder_path']}) on engine {item['engine_version']}: "
            f"{item['total_assets']} assets, {item['map_count']} maps, "
            f"{item['environment_assets']} environment assets, "
            f"{item['character_blueprints']} character blueprints, "
            f"{item['pawn_blueprints']} pawn blueprints, "
            f"{item['skeletal_meshes']} skeletal meshes, "
            f"{item['material_count']} materials, "
            f"{item['niagara_systems']} Niagara systems."
        )
    lines.append(
        "Tip: import richer Unreal metadata JSON via /unreal/import for best "
        "character/environment detail."
    )


def _append_unreal_profile_hint(lines: List[str], unreal_profiles: List[Dict[str, Any]]) -> None:
    detected = ", ".join(profile.get("folder_tag", "Unknown") for profile in unreal_profiles[:4])
    lines.append(
        "Detected Unreal project(s): "
        f"{detected}. Would you like me to run deeper Unreal analysis "
        "for environment, characters, maps, and gameplay assets?"
    )
    lines.append(
        "To enable this, import Unreal metadata JSON with POST /unreal/import "
        "(json_path + optional folder_tag)."
    )


def _append_project_profile_lines(lines: List[str], folder_profiles: List[Dict[str, Any]]) -> None:
    lines.append("Indexed project/folder profiles:")
    for profile in folder_profiles[:8]:
        size_mb = round((profile.get("total_size_bytes", 0) or 0) / (1024 * 1024), 2)
        lines.append(
            f"- {profile['folder_tag']} ({profile['project_type']}): "
            f"{profile['file_count']} files, {size_mb} MB, path: {profile['folder_path']}"
        )


def _append_inventory_type_lines(lines: List[str], file_stats: Dict[str, Any]) -> None:
    top_types = file_stats.get("by_type", [])[:6]
    if not top_types:
        return
    lines.append(
        "Top file types: "
        + ", ".join(f"{item['ext']} ({item['count']})" for item in top_types)
    )


def _build_fast_answer(
    query: str,
    file_stats: Optional[Dict[str, Any]],
    folder_profiles: List[Dict[str, Any]],
    unreal_facts: List[Dict[str, Any]],
) -> Optional[str]:
    """Build a deterministic answer for inventory/project questions.

    This bypasses slow LLM calls when structured metadata is sufficient.
    """
    inventory = _is_inventory_query(query)
    project = _is_project_query(query)
    unreal = _is_unreal_query(query)

    if not (inventory or project or unreal):
        return None

    lines: List[str] = []
    unreal_profiles = [
        profile for profile in folder_profiles
        if "unreal" in str(profile.get("project_type", "")).lower()
    ]

    if file_stats:
        lines.append(
            f"You currently have {file_stats['total_files']} indexed files "
            f"(~{file_stats['total_size_mb']} MB)."
        )

    if unreal and unreal_facts:
        _append_unreal_fact_lines(lines, unreal_facts)

    if unreal_profiles and not unreal_facts:
        _append_unreal_profile_hint(lines, unreal_profiles)

    if project and folder_profiles:
        _append_project_profile_lines(lines, folder_profiles)

    if inventory and file_stats:
        _append_inventory_type_lines(lines, file_stats)

    if not lines:
        return None
    return "\n".join(lines)

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
    use_reranker: bool = True,
) -> List[Dict[str, Any]]:
    """Combines FTS5 keyword + Chroma semantic + document-summary search using RRF,
    then reranks the candidates with a cross-encoder for maximum precision."""

    query_emb = await embedding_service.embed_query(query)

    fts_results, semantic_results, relevant_doc_paths = await asyncio.gather(
        _fts_search(db, query, k),
        _semantic_search_with_emb(chroma_client, query_emb, k),
        _summary_search_with_emb(chroma_client, query_emb, k),
    )

    sorted_ids = _compute_rrf_scores(fts_results, semantic_results, k * 2)

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

    results = []
    for cid in chunk_ids_ordered:
        row = row_map.get(cid)
        if row:
            file_path = row[2]
            rrf_score = score_map[cid] * settings.rrf_score_scale
            if file_path in relevant_doc_paths:
                rrf_score *= settings.summary_boost_factor
            results.append({
                "chunk_id": cid,
                "text": row[1],
                "file_path": file_path,
                "folder_tag": row[3],
                "score": round(rrf_score, 4),
            })

    if results and use_reranker:
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
    For inventory-type questions augments context with aggregate file stats.
    For project-level questions augments context with folder profiles.
    """
    t_start = time.perf_counter()

    inventory = _is_inventory_query(query)
    project = _is_project_query(query)
    unreal = _is_unreal_query(query)

    folder_profiles, file_stats, unreal_facts = await _load_query_metadata(
        db,
        inventory=inventory,
        project=project,
        unreal=unreal,
    )

    fast_answer = _build_fast_answer(query, file_stats, folder_profiles, unreal_facts)
    if fast_answer:
        source_rows = [
            {
                "file_path": p.get("folder_path", ""),
                "folder_tag": p.get("folder_tag", ""),
                "text": p.get("profile_text", ""),
            }
            for p in folder_profiles
        ]
        return {
            "answer": fast_answer,
            "sources": source_rows,
            "retrieved_count": len(source_rows),
            "latency_ms": round((time.perf_counter() - t_start) * 1000, 1),
        }

    # Fallback to full RAG for semantic questions.
    include_profiles_text = project or inventory or unreal
    retrieved, file_stats, folder_profiles_text = await _gather_full_rag_inputs(
        query=query,
        db=db,
        embedding_service=embedding_service,
        chroma_client=chroma_client,
        k=k,
        inventory=inventory,
        project=project,
        unreal=unreal,
        cached_file_stats=file_stats,
        include_profiles_text=include_profiles_text,
    )

    t_retrieval = time.perf_counter()

    if file_type or folder_tag:
        retrieved = _filter_retrieved_results(retrieved, file_type=file_type, folder_tag=folder_tag)
    
    if not retrieved and not file_stats and not folder_profiles_text:
        return {
            "answer": "I couldn't find any relevant documents to answer your question.",
            "sources": [],
            "retrieved_count": 0,
            "latency_ms": round((time.perf_counter() - t_start) * 1000, 1),
        }
        
    context = build_context(
        retrieved,
        file_stats=file_stats,
        folder_profiles_text=folder_profiles_text,
    )
    t_context = time.perf_counter()
    
    answer = await llm_client.generate_answer(query, context)
    t_llm = time.perf_counter()
    
    total_ms = round((t_llm - t_start) * 1000, 1)
    logger.info(
        "RAG pipeline: retrieval=%.0fms context=%.0fms llm=%.0fms total=%.0fms "
        "(inventory=%s project=%s unreal=%s)",
        (t_retrieval - t_start) * 1000,
        (t_context - t_retrieval) * 1000,
        (t_llm - t_context) * 1000,
        total_ms,
        inventory,
        project,
        unreal,
    )
    
    return {
        "answer": answer,
        "sources": retrieved,
        "retrieved_count": len(retrieved),
        "latency_ms": total_ms,
    }


async def _load_query_metadata(
    db: DatabaseManager,
    inventory: bool,
    project: bool,
    unreal: bool,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    should_load_profiles = project or inventory or unreal
    should_load_unreal_facts = unreal or project

    folder_profiles_coro = (
        db.get_all_folder_profiles()
        if should_load_profiles
        else asyncio.sleep(0, result=[])
    )
    file_stats_coro = db.get_file_stats_summary() if inventory else asyncio.sleep(0, result=None)
    unreal_facts_coro = (
        db.get_all_unreal_project_facts()
        if should_load_unreal_facts
        else asyncio.sleep(0, result=[])
    )

    return await asyncio.gather(folder_profiles_coro, file_stats_coro, unreal_facts_coro)


async def _gather_full_rag_inputs(
    query: str,
    db: DatabaseManager,
    embedding_service: EmbeddingService,
    chroma_client: ChromaClient,
    k: int,
    inventory: bool,
    project: bool,
    unreal: bool,
    cached_file_stats: Optional[Dict[str, Any]],
    include_profiles_text: bool,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    coros: List[Any] = [
        hybrid_retrieve(
            query=query,
            db=db,
            embedding_service=embedding_service,
            chroma_client=chroma_client,
            k=k,
            use_reranker=not (project or inventory or unreal),
        )
    ]

    if inventory:
        coros.append(asyncio.sleep(0, result=cached_file_stats))
    if include_profiles_text:
        coros.append(db.get_folder_profiles_text())

    results = await asyncio.gather(*coros)

    retrieved = results[0]
    file_stats = cached_file_stats if inventory else None
    folder_profiles_text = ""
    next_idx = 1

    if inventory:
        file_stats = results[next_idx]
        next_idx += 1
    if include_profiles_text:
        folder_profiles_text = results[next_idx]

    return retrieved, file_stats, folder_profiles_text


def _filter_retrieved_results(
    retrieved: List[Dict[str, Any]],
    file_type: Optional[str],
    folder_tag: Optional[str],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for result in retrieved:
        file_path = result.get("file_path", "")
        result_folder_tag = result.get("folder_tag", "")
        if file_type and not file_path.lower().endswith(file_type):
            continue
        if folder_tag and result_folder_tag != folder_tag:
            continue
        filtered.append(result)
    return filtered
