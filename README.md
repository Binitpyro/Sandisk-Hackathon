# Personal Memory Assistant

A local‑first Personal Memory Assistant that helps users rediscover their own work.

Most people have documents, PDFs, source code and project reports scattered across laptops, external drives and random folders. When they later need something like “my feedback for my operating systems project” or “the PDF for my internship offer”, they end up clicking through directories or trying to remember file names.

This project indexes the user’s chosen folders once, keeps that index up to date, and lets the user ask natural‑language questions over their own files. The goal is to let users treat their local storage as a “second memory”.

## Tech Stack

- **Backend:** Python 3.12 + FastAPI
- **Server:** Uvicorn (ASGI)
- **Storage:**
  - SQLite (+ FTS5) for metadata and keyword search
  - Chroma (on‑disk) for vector search
- **ML:**
  - `sentence-transformers` for local embeddings
  - Pluggable LLM (local or cloud) for answer generation
- **UI:** Simple HTML/JS served by FastAPI

## Running (placeholder)

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows

pip install -r requirements.txt

uvicorn app.main:app --reload
```

See `docs/ARCHITECTURE.md` and `docs/IMPLEMENTATION_PLAN.md` for design and plan.
