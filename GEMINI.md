# Gemini Context – Personal Memory Assistant

You are an AI pair‑programmer helping build a **local‑first Personal Memory Assistant**.

Always read these files before making design or code changes:

- @./README.md
- @./docs/ARCHITECTURE.md
- @./docs/IMPLEMENTATION_PLAN.md
- @./docs/API_SPEC.md
- @./docs/INDEXING_AND_RAG.md

## Project Summary

- Index user‑selected local folders (Documents, Projects, Academic Work, etc.).
- Extract text from supported files and build:
  - A metadata index in SQLite (with FTS5) for keyword and structured search.
  - A vector index in Chroma for semantic search.
- Provide a small UI where the user can ask questions in natural language.
- Return answers plus references to the original files/snippets.
- Run entirely on the user’s machine, staying lightweight and privacy‑preserving.

## Coding Guidelines

- Language: Python 3.12.
- Web framework: **FastAPI** with async endpoints where I/O‑bound.
- Use **SQLite via `aiosqlite`** and **Chroma’s Python client** directly (no LangChain).
- Keep functions small and focused; prefer dependency injection (pass DB/Chroma clients in).
- Aim for clear, readable code; add docstrings for public functions.

## How I Want You To Help

When I ask questions or request code:

1. **Prefer editing existing files over large rewrites.**
2. Propose **concrete file paths** and function names (e.g. `app/indexing/service.py::index_folder`).
3. When suggesting new features:
   - Update or reference the relevant doc (`ARCHITECTURE`, `API_SPEC`, or `IMPLEMENTATION_PLAN`).
4. Assume a single‑user, local app – no multi‑tenant or heavy infra.
5. If something is ambiguous, ask me to choose between 2–3 concrete options.

## Non‑Goals (for now)

- No LangChain, no heavy orchestration frameworks.
- No external databases (no Postgres, no managed vector DBs).
- No user accounts, auth, or multi‑user features.
- No desktop packaging (Electron, etc.) in the hackathon phase.
