from typing import List, Dict, Any, Optional
from app.config import settings

def _format_file_stats(stats: Dict[str, Any]) -> str:
    """Format aggregate file statistics into a readable preamble for the LLM."""
    lines = [
        "=== File Statistics (from your indexed files) ===",
        f"Total indexed files: {stats['total_files']}",
        f"Total size: {stats['total_size_mb']} MB",
        "",
        "Files by type:",
    ]
    for t in stats["by_type"]:
        lines.append(f"  {t['ext']}: {t['count']} files ({t['size_mb']} MB)")
    lines.append("")
    lines.append("Files by folder/project:")
    for f in stats["by_folder"]:
        lines.append(f"  {f['folder']}: {f['count']} files")
    lines.append("=" * 50)
    return "\n".join(lines)


def build_context(
    retrieved_results: List[Dict[str, Any]],
    max_tokens: int = 0,
    file_stats: Optional[Dict[str, Any]] = None,
    folder_profiles_text: str = "",
) -> str:
    """Formats retrieved snippets into a single context string for the LLM.

    If *file_stats* is provided (aggregate counts by type/folder), it is
    prepended so the LLM can answer inventory questions with hard numbers
    instead of listing every individual file.

    If *folder_profiles_text* is provided (project-level descriptions), it
    is prepended so the LLM can answer questions about what projects exist,
    their types, and their structure.
    """
    if not retrieved_results and not file_stats and not folder_profiles_text:
        return "No relevant context found."

    if max_tokens <= 0:
        max_tokens = settings.context_max_tokens

    context_parts = []
    total_len = 0
    max_chars = max_tokens * 4  # ~4 chars per token heuristic

    # Prepend folder profiles when available (project-level understanding)
    if folder_profiles_text:
        context_parts.append(folder_profiles_text)
        total_len += len(folder_profiles_text)

    # Prepend file statistics when available
    if file_stats:
        stats_block = _format_file_stats(file_stats)
        context_parts.append(stats_block)
        total_len += len(stats_block)
    
    for i, res in enumerate(retrieved_results):
        snippet_id = i + 1
        path = res.get("file_path", "Unknown File")
        text = res.get("text", "")
        
        part = f"""Snippet {snippet_id} [{path}]:
{text}
---
"""
        if total_len + len(part) > max_chars:
            break
            
        context_parts.append(part)
        total_len += len(part)
            
    return "\n".join(context_parts)
