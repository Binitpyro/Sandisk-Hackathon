-- Files table for storing metadata
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    size INTEGER NOT NULL,
    modified_at TEXT NOT NULL,
    type TEXT NOT NULL,
    folder_tag TEXT,
    usage_count INTEGER DEFAULT 0
);

-- Chunks table for storing text segments
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    text_preview TEXT NOT NULL,
    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
);

-- FTS5 virtual table for keyword search
-- Note: SQLite FTS5 should be enabled in the environment
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    chunks_text,
    content=chunks,
    content_rowid=id
);

-- Trigger to keep FTS index in sync with chunks table
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunk_fts(rowid, chunks_text) VALUES (new.id, new.text_preview);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunk_fts(chunk_fts, rowid, chunks_text) VALUES('delete', old.id, old.text_preview);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunk_fts(chunk_fts, rowid, chunks_text) VALUES('delete', old.id, old.text_preview);
  INSERT INTO chunk_fts(rowid, chunks_text) VALUES (new.id, new.text_preview);
END;
