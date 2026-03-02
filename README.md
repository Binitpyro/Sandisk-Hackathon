# Personal Memory Assistant 🧠

A local-first Personal Memory Assistant that helps you rediscover your own work across files (TXT, MD, PDF).

Built with **Python 3.12**, **FastAPI**, **SQLite (FTS5)**, **ChromaDB**, and **Sentence-Transformers**.

## Features

- **Hybrid Search**: Combines keyword search (FTS5) and semantic search (Vector) for better results.
- **Local-First**: All metadata and embeddings are stored on your machine.
- **LLM-Powered Answers**: Uses Gemini 1.5 Flash (or local Ollama) to generate natural language answers with citations.
- **Storage Insights**: Identify "cold" files that take up space but haven't been relevant to your queries.
- **Modern UI**: Clean, tabbed interface for indexing, querying, and insights.

## Quick Start

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 2. Configure LLM (Optional but Recommended)
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_api_key_here
```
*If no key is provided, the app will attempt to fallback to a local Ollama instance at `http://localhost:11434`.*

### 3. Run the App
```bash
uvicorn app.main:app --reload
```
Visit `http://localhost:8000` in your browser.

## How to Use

1. **Index**: Go to the **Index** tab. You can paste absolute folder paths or click **Seed Demo Data** to quickly generate and index sample files.
2. **Ask**: Go to the **Ask** tab. Type a question like "What feedback did I get on my project?" or "When does my internship start?".
3. **Insights**: Visit the **Insights** tab to see your storage breakdown and find files that are rarely used.

## Unreal Deep Metadata (Environment / Characters / Assets)

For Unreal projects, binary assets (`.uasset`, `.umap`) cannot be deeply parsed as plain text.
To get rich project understanding, import Unreal metadata JSON:

```bash
curl -X POST http://127.0.0.1:8000/unreal/import \
	-H "Content-Type: application/json" \
	-d '{"json_path":"C:/path/to/unreal_metadata.json","folder_tag":"MyProject"}'
```

After import, queries like "describe the unreal project", "how many characters/maps", or
"what environment assets do I have" use structured facts and return much faster.

## Query Latency Optimization

- Inventory/project questions now use a deterministic fast path (no LLM call).
- Default retrieval settings were reduced for lower latency (`retrieval_top_k=6`, `context_max_tokens=2200`).
- Full LLM RAG is still used for deep semantic content questions.

## Architecture

- **Backend**: FastAPI (Python)
- **Metadata**: SQLite with FTS5 for lightning-fast keyword searches.
- **Vector Store**: ChromaDB for semantic embeddings.
- **Embeddings**: Local `all-MiniLM-L6-v2` (Sentence-Transformers).
- **RAG Engine**: Reciprocal Rank Fusion (RRF) to merge keyword and semantic results.
