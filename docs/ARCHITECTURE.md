# Architecture

This document describes the system architecture of the Personal Memory Assistant.

## 1. High‑Level Views

The system has five main areas:

1. **User & UI**
2. **Indexing Service**
3. **Embedding & Storage**
4. **Query & RAG Engine**
5. **Storage Insights**

These roughly follow a layered, scenario‑driven view (separating concerns by stakeholder and use‑case).

---

## 2. Components

### 2.1 User & UI

- Simple HTML/JS frontend served by FastAPI.
- Pages:
  - Folder selection + indexing status.
  - “Ask a question” view.
  - Optional “Storage insights” view.

### 2.2 Indexing Service

Responsibilities:

- Perform an initial recursive scan over user‑selected folders.
- Apply ignore rules (build folders, caches, very large binaries).
- For each supported file:
  - Extract plain text (TXT/MD directly; PDFs via parser).
  - Chunk text into overlapping segments.

Exposed as:

- Background task started at app boot or via `/index/start` endpoint.

### 2.3 Embedding & Storage

**Embedding Service**

- Uses a small, local `sentence-transformers` model to generate embeddings.
- Batches inputs to improve throughput.

**Metadata DB (SQLite + FTS5)**

- Tables:
  - `files(id, path, size, modified_time, type, folder_tag, ...)`
  - `chunks(id, file_id, start_offset, end_offset, text_preview, created_at)`
- FTS5 virtual table:
  - `chunk_search(chunk_id, text)` for keyword search.

**Vector Store (Chroma)**

- One collection, e.g. `pma_chunks`.
- Each record: embedding + `chunk_id` and metadata (file path, folder tag).

### 2.4 Query & RAG Engine

Flow:

1. User submits a natural‑language question.
2. **Query Handler** normalises input and applies filters.
3. **Hybrid Retrieval**:
   - Keyword search via SQLite FTS5.
   - Semantic search via Chroma.
   - Merge + deduplicate results.
4. **Context Builder** selects top‑K snippets and assembles context.
5. **LLM Answer Generator** calls a pluggable LLM (local or cloud) with:
   - Question
   - Snippets + file metadata
6. Response returned to UI:
   - Short answer
   - List of sources (file path + snippet).

### 2.5 Storage Insights

- Periodically aggregates from SQLite:
  - Total indexed size.
  - Largest N files.
  - “Cold” large files (large but never in query results).

---

## 3. Cross‑Cutting Concerns

- **Privacy:** All file contents and indexes stored locally on disk.
- **Performance:** Index once, then incremental updates; on‑disk vector store to limit RAM.
- **Observability:** Basic logging around indexing runs, query latency, and errors.

See `INDEXING_AND_RAG.md` for detailed indexing and retrieval behaviour.
