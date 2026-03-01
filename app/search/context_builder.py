from typing import List, Dict, Any
from app.config import settings


def build_context(retrieved_results: List[Dict[str, Any]], max_tokens: int = 0) -> str:
    """Formats retrieved snippets into a single context string for the LLM."""
    if not retrieved_results:
        return "No relevant context found."

    if max_tokens <= 0:
        max_tokens = settings.context_max_tokens

    context_parts = []
    total_len = 0
    max_chars = max_tokens * 4  # ~4 chars per token heuristic
    
    for i, res in enumerate(retrieved_results):
        snippet_id = i + 1
        path = res.get("file_path", "Unknown File")
        text = res.get("text", "")
        
        # Format the snippet with its source
        part = f"""Snippet {snippet_id} [{path}]:
{text}
---
"""
        if total_len + len(part) > max_chars:
            break
            
        context_parts.append(part)
        total_len += len(part)
            
    return "\n".join(context_parts)
