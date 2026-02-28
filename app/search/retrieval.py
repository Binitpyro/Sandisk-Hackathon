import logging
from typing import List, Dict, Any, Optional
from app.storage.db import DatabaseManager
from app.embeddings.service import EmbeddingService
from app.vector_store.chroma_client import ChromaClient
from app.search.context_builder import build_context
from app.search.llm_client import LLMClient

logger = logging.getLogger(__name__)

async def hybrid_retrieve(
    query: str, 
    k: int = 5, 
    db: DatabaseManager = None, 
    embedding_service: EmbeddingService = None, 
    chroma_client: ChromaClient = None
) -> List[Dict[str, Any]]:
    """Combines FTS5 keyword search and Chroma semantic search using RRF."""
    
    # 1. Keyword Search (SQLite FTS5)
    fts_query = "SELECT rowid, chunks_text FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY rank LIMIT ?"
    fts_results = []
    try:
        async with db.conn.execute(fts_query, (query, 2 * k)) as cursor:
            rows = await cursor.fetchall()
            fts_results = [{"id": str(row[0]), "text": row[1]} for row in rows]
    except Exception as e:
        logger.error(f"FTS5 Search failed: {e}")

    # 2. Semantic Search (Chroma)
    query_emb = await embedding_service.embed_query(query)
    semantic_results_raw = await chroma_client.semantic_search(query_emb, k=2 * k)
    
    semantic_results = []
    if semantic_results_raw["ids"] and semantic_results_raw["ids"][0]:
        for i in range(len(semantic_results_raw["ids"][0])):
            semantic_results.append({
                "id": semantic_results_raw["ids"][0][i],
                "score": semantic_results_raw["distances"][0][i] if "distances" in semantic_results_raw else 0.0
            })

    # 3. Reciprocal Rank Fusion (RRF)
    scores = {}
    K_RRF = 60
    
    for rank, res in enumerate(fts_results):
        chunk_id = res["id"]
        score = 0.4 * (1.0 / (K_RRF + rank + 1))
        scores[chunk_id] = score
        
    for rank, res in enumerate(semantic_results):
        chunk_id = res["id"]
        score = 0.6 * (1.0 / (K_RRF + rank + 1))
        scores[chunk_id] = scores.get(chunk_id, 0) + score

    sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    # 4. Fetch full metadata and text from DB
    results = []
    for chunk_id, score in sorted_ids:
        query_sql = """
        SELECT c.text_preview, f.path, f.folder_tag 
        FROM chunks c 
        JOIN files f ON c.file_id = f.id 
        WHERE c.id = ?
        """
        async with db.conn.execute(query_sql, (int(chunk_id),)) as cursor:
            row = await cursor.fetchone()
            if row:
                results.append({
                    "chunk_id": int(chunk_id),
                    "text": row[0],
                    "file_path": row[1],
                    "folder_tag": row[2],
                    "score": round(score * 1000, 4)
                })
                
    return results

async def full_rag(
    query: str,
    db: DatabaseManager,
    embedding_service: EmbeddingService,
    chroma_client: ChromaClient,
    llm_client: LLMClient,
    k: int = 5
) -> Dict[str, Any]:
    """Retrieves context and generates an LLM answer."""
    
    # Step 1: Hybrid Retrieval
    retrieved = await hybrid_retrieve(
        query=query,
        k=k,
        db=db,
        embedding_service=embedding_service,
        chroma_client=chroma_client
    )
    
    if not retrieved:
        return {
            "answer": "I couldn't find any relevant documents to answer your question.",
            "sources": [],
            "retrieved_count": 0
        }
        
    # Step 2: Build Context
    context = build_context(retrieved)
    
    # Step 3: LLM Generation
    answer = await llm_client.generate_answer(query, context)
    
    # Step 4: Parse answer and return
    return {
        "answer": answer,
        "sources": retrieved,
        "retrieved_count": len(retrieved)
    }
