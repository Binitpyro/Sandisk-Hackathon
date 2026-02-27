# Indexing and RAG Details

This doc captures the low‑level behaviour of indexing, chunking, retrieval and answer generation.

## 1. Indexing Strategy

- **Initial full scan** over user‑selected roots.
- Skip:
  - Hidden/system folders
  - Known build directories (`node_modules`, `bin`, `obj`, `dist`, etc.)
  - Very large binaries (configurable size threshold)
- Supported file types (MVP):
  - `.txt`, `.md`, `.pdf` (text‑only at first)
- For each file:
  - Extract text.
  - Chunk into segments of ~300–500 tokens with ~50–100 token overlap.
  - Store:
    - File row in SQLite `files`.
    - Chunk row in `chunks` (+ entry in FTS5 table).
    - Embedding in Chroma with metadata `{chunk_id, file_path, folder_tag}`.

## 2. Incremental Updates

- Use OS file watching where possible.
- On file create/modify:
  - Re‑index only that file.
- On delete:
  - Remove its rows from SQLite and Chroma.
- Keep stats on:
  - Last index run
  - Number of files/chunks
  - Average per‑file indexing time

## 3. Retrieval

### Keyword

- Query FTS5 table with the user’s question (and/or extracted keywords).
- Get top N matching `chunk_id`s.

### Semantic

- Embed the question with the same model as chunks.
- Query Chroma for top M nearest neighbours.

### Hybrid Merge

- Combine candidates from both sources.
- Score function can be a weighted sum of:
  - FTS relevance
  - Vector similarity
- Deduplicate on `chunk_id` and keep best score.

## 4. Answer Generation

- Take top‑K chunks after merging (e.g. 5–10).
- Build a prompt for the LLM:

  - System: "You are a personal memory assistant that answers questions using only the provided user files."
  - Context: snippets + file paths
  - User: question

- Ask for:
  - A short, direct answer
  - A bullet list of cited files

If no LLM is configured, return the merged snippets and sources directly.

## 5. Quality Considerations

- Try to keep chunks self‑contained and semantically coherent; avoids "half‑sentences" in answers.
- Prefer clear headings and summaries in docs; improves embedding and retrieval quality.
- Avoid indexing huge monolithic files if they mix unrelated topics; consider splitting them logically when possible.
