from typing import List, Dict, Any

def build_context(retrieved_results: List[Dict[str, Any]], max_tokens: int = 2000) -> str:
    """Formats retrieved snippets into a single context string for the LLM."""
    if not retrieved_results:
        return "No relevant context found."

    context_parts = []
    
    for i, res in enumerate(retrieved_results):
        snippet_id = i + 1
        path = res.get("file_path", "Unknown File")
        text = res.get("text", "")
        
        # Format the snippet with its source
        part = f"""Snippet {snippet_id} [{path}]:
{text}
---
"""
        context_parts.append(part)
        
        # Roughly check token count (4 chars per token heuristic)
        total_len = sum(len(p) for p in context_parts)
        if total_len > (max_tokens * 4):
            context_parts.pop()
            break
            
    return "\n".join(context_parts)
