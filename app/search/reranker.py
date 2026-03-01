import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder  # ships with sentence-transformers

logger = logging.getLogger(__name__)

_reranker: Optional[CrossEncoder] = None
_reranker_lock = threading.Lock()
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def _get_model() -> CrossEncoder:
    """Lazily load the cross-encoder (≈ 80 MB, ~120 ms on CPU). Thread-safe."""
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:  # double-checked locking
                logger.info("Loading reranker model: %s", _MODEL_NAME)
                _reranker = CrossEncoder(_MODEL_NAME, max_length=512)
                logger.info("Reranker model loaded.")
    return _reranker

async def rerank(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int = 10,
    text_key: str = "text",
) -> List[Dict[str, Any]]:
    """Re-score *results* against *query* and return the top-k by relevance.

    Each item in *results* must contain a ``text_key`` field that holds the
    chunk text.  A ``rerank_score`` field is added to every returned item.

    Parameters
    ----------
    query:
        The user's natural-language question.
    results:
        Candidate chunks produced by the hybrid retriever.
    top_k:
        How many results to keep after reranking.
    text_key:
        Key in each result dict that contains the text to score.

    Returns
    -------
    list:
        The *top_k* results sorted by descending cross-encoder score.
    """
    if not results:
        return results

    loop = asyncio.get_running_loop()
    # Load model off the event loop to avoid blocking during first download/load
    model = await loop.run_in_executor(None, _get_model)
    pairs = [(query, item[text_key]) for item in results]

    scores = await loop.run_in_executor(
        None,
        lambda: model.predict(pairs, show_progress_bar=False).tolist(),
    )

    for item, score in zip(results, scores):
        item["rerank_score"] = round(float(score), 6)

    ranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    top = ranked[:top_k]

    logger.debug(
        "Reranked %d candidates → top-%d (best=%.4f, worst=%.4f)",
        len(results),
        top_k,
        top[0]["rerank_score"] if top else 0,
        top[-1]["rerank_score"] if top else 0,
    )
    return top
