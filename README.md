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

## Architecture

- **Backend**: FastAPI (Python)
- **Metadata**: SQLite with FTS5 for lightning-fast keyword searches.
- **Vector Store**: ChromaDB for semantic embeddings.
- **Embeddings**: Local `all-MiniLM-L6-v2` (Sentence-Transformers).
- **RAG Engine**: Reciprocal Rank Fusion (RRF) to merge keyword and semantic results.
