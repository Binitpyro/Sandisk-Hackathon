# Implementation Plan

This is a practical, step‑by‑step plan for building the MVP.

## Milestone 0 – Setup

- [ ] Create `.venv`, install dependencies (`fastapi`, `uvicorn[standard]`, `aiosqlite`, `chromadb`, `sentence-transformers`, `python-dotenv`, `jinja2`, `httpx`).
- [ ] Add `.gitignore`, `README.md`, `GEMINI.md`, and docs in `docs/`.

## Milestone 1 – API Skeleton

- [ ] Create `app/main.py` with FastAPI app.
- [ ] Add health endpoint: `GET /health`.
- [ ] Configure static files/templates for simple HTML UI.

## Milestone 2 – Metadata DB and Indexing

- [ ] Define SQLite schema in `app/storage/schema.sql`.
- [ ] Implement DB helper in `app/storage/db.py` (init DB, migrations, basic CRUD).
- [ ] Implement `app/indexing/service.py`:
  - `index_folders(folders: list[Path])`
  - `index_file(path: Path)`
- [ ] Wire basic indexing to `POST /index/start`.

## Milestone 3 – Embeddings and Chroma

- [ ] Create `app/embeddings/service.py`:
  - Load `sentence-transformers` model.
  - Batch `embed_texts(texts: list[str])`.
- [ ] Create `app/vector_store/chroma_client.py`:
  - Manage collection, add/update/delete documents.
- [ ] Integrate embedding + Chroma calls into indexing pipeline.

## Milestone 4 – Query & Hybrid Retrieval

- [ ] Implement `app/search/retrieval.py`:
  - Keyword search via FTS5.
  - Semantic search via Chroma.
  - Merge + score results.
- [ ] Implement `app/search/context_builder.py`.
- [ ] Implement `app/search/llm_client.py` (initially stubbed; can return top‑K snippets if no LLM configured).
- [ ] Expose `POST /query` endpoint.

## Milestone 5 – UI and Demo

- [ ] Basic HTML page with:
  - Folder input text box or picker.
  - Button to kick off indexing.
  - Textarea or input for query.
  - Area to show answer and sources.
- [ ] Seed some sample folders and test queries.
- [ ] Capture screenshots for slides.
