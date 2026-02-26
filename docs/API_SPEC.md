# API Specification (Draft)

Base URL: `http://localhost:8000`

## Health

### GET `/health`

- **Response 200:**
  - `{ "status": "ok" }`

---

## Indexing

### POST `/index/start`

Trigger (re)indexing over a set of folders.

- **Request body (JSON):**
  ```json
  {
    "folders": ["D:/Docs", "D:/Projects"]
  }
  ```

- **Response 202:**
  ```json
  { "message": "Indexing started", "job_id": "<id>" }
  ```

### GET `/index/status`

Get high‑level indexing status.

- **Response 200:**
  ```json
  {
    "status": "idle" | "running" | "error",
    "files_indexed": 1234,
    "chunks_indexed": 9876,
    "last_run": "2025-03-01T10:30:00Z"
  }
  ```

---

## Query

### POST `/query`

Ask a natural‑language question.

- **Request body (JSON):**
  ```json
  {
    "question": "What feedback did I get on my OS lab project?",
    "filters": {
      "folders": ["Academic Work"],
      "file_types": ["pdf", "md"]
    }
  }
  ```

- **Response 200:**
  ```json
  {
    "answer": "You received feedback about improving error handling in the process scheduler.",
    "sources": [
      {
        "file_path": "D:/Academic/OS_Lab/feedback.pdf",
        "snippet": "The TA suggested adding better error handling around ...",
        "score": 0.92
      }
    ]
  }
  ```

If no LLM is configured, `answer` can be a simple summary like “Top 3 relevant snippets found” and `sources` still populated.

---

## Insights (Optional)

### GET `/insights/storage`

Summary of indexed storage.

- **Response 200:**
  ```json
  {
    "total_size_bytes": 123456789,
    "file_count": 4321,
    "largest_files": [
      { "path": "...", "size_bytes": 12345678 },
      { "path": "...", "size_bytes": 9876543 }
    ],
    "cold_large_files": [
      { "path": "...", "size_bytes": 11111111 }
    ]
  }
  ```
