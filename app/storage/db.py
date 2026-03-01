"""
Database manager module for Personal Memory Assistant.
Handles interactions with SQLite using aiosqlite for metadata storage.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages the SQLite database connection and operations."""

    def __init__(self, db_path: str = "pma_metadata.db"):
        """Initializes the DatabaseManager."""
        self.db_path = db_path
        self.conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Establish connection to the SQLite database."""
        if not self.conn:
            self.conn = await aiosqlite.connect(self.db_path)
            self.conn.row_factory = aiosqlite.Row
            await self.conn.execute("PRAGMA journal_mode = WAL;")
            await self.conn.execute("PRAGMA foreign_keys = ON;")
            await self.conn.execute("PRAGMA synchronous = NORMAL;")
            await self.conn.execute("PRAGMA busy_timeout = 5000;")

    def _get_conn(self) -> aiosqlite.Connection:
        """Return the active connection, raising if not connected."""
        if self.conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.conn

    async def close(self):
        """Close the active database connection if open."""
        if self.conn:
            await self.conn.close()
            self.conn = None

    async def init_db(self, schema_path: str = "app/storage/schema.sql") -> None:
        """Initialize the database with the schema."""
        await self.connect()
        conn = self._get_conn()
        try:
            schema = Path(schema_path).read_text(encoding="utf-8")
            await conn.executescript(schema)
            await conn.commit()
            logger.info("Database initialized successfully.")
        except Exception as e:
            logger.error("Error initializing database: %s", e)
            raise
        await self._migrate(conn)

    async def _migrate(self, conn: "aiosqlite.Connection") -> None:
        """Apply safe, idempotent schema migrations."""
        migrations = [
            ("summary", "ALTER TABLE files ADD COLUMN summary TEXT DEFAULT ''"),
            ("files_created_at",
             "ALTER TABLE files ADD COLUMN created_at TEXT NOT NULL DEFAULT ''"),
            ("chunks_created_at",
             "ALTER TABLE chunks ADD COLUMN created_at TEXT NOT NULL DEFAULT ''"),
        ]
        for col_name, ddl in migrations:
            try:
                await conn.execute(ddl)
                await conn.commit()
                logger.info("Migration applied: added column '%s'.", col_name)
            except Exception as exc:
                if "duplicate column" in str(exc).lower():
                    logger.debug("Column '%s' already exists — skipping.", col_name)
                else:
                    logger.error("Migration failed for column '%s': %s", col_name, exc)
                    raise

    async def insert_file(self, file_data: Dict[str, Any]) -> int:
        """Inserts file metadata and returns the new file id."""
        conn = self._get_conn()
        file_data.setdefault("summary", "")
        query = """
        INSERT INTO files (path, size, modified_at, type, folder_tag, summary)
        VALUES (:path, :size, :modified_at, :type, :folder_tag, :summary)
        ON CONFLICT(path) DO UPDATE SET
            size=excluded.size,
            modified_at=excluded.modified_at,
            type=excluded.type,
            folder_tag=excluded.folder_tag,
            summary=excluded.summary
        RETURNING id;
        """
        async with conn.execute(query, file_data) as cursor:
            row = await cursor.fetchone()
            if row is None:
                raise RuntimeError(f"INSERT RETURNING id failed for {file_data.get('path')}")
            file_id: int = row[0]
            await conn.commit()
            return file_id

    async def insert_chunk(self, chunk_data: Dict[str, Any]) -> int:
        """Inserts a chunk and returns the new chunk id."""
        conn = self._get_conn()
        query = """
        INSERT INTO chunks (file_id, start_offset, end_offset, text_preview)
        VALUES (:file_id, :start_offset, :end_offset, :text_preview)
        RETURNING id;
        """
        async with conn.execute(query, chunk_data) as cursor:
            row = await cursor.fetchone()
            if row is None:
                raise RuntimeError("INSERT RETURNING id failed for chunk")
            chunk_id: int = row[0]
            return chunk_id

    async def insert_chunks_bulk(self, chunks: List[Dict[str, Any]]) -> List[int]:
        """Insert multiple chunks in a single transaction using executemany.

        Returns a list of inserted chunk IDs.
        """
        if not chunks:
            return []
        conn = self._get_conn()
        ids: List[int] = []
        for chunk in chunks:
            async with conn.execute(
                "INSERT INTO chunks (file_id, start_offset, end_offset, text_preview) "
                "VALUES (:file_id, :start_offset, :end_offset, :text_preview) RETURNING id;",
                chunk,
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    ids.append(row[0])
        return ids

    async def commit(self) -> None:
        """Explicitly commits the current transaction."""
        if self.conn:
            await self.conn.commit()

    async def delete_file_chunks(self, file_id: int) -> None:
        """Deletes all chunks associated with a file."""
        conn = self._get_conn()
        await conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        await conn.commit()

    async def get_file_chunks(self, file_id: int) -> List[aiosqlite.Row]:
        """Returns all chunks for a given file id."""
        conn = self._get_conn()
        async with conn.execute("SELECT * FROM chunks WHERE file_id = ?", (file_id,)) as cursor:
            return list(await cursor.fetchall())

    async def get_file_by_path(self, path: str) -> Optional[aiosqlite.Row]:
        """Returns file metadata by path."""
        conn = self._get_conn()
        async with conn.execute("SELECT * FROM files WHERE path = ?", (path,)) as cursor:
            return await cursor.fetchone()

    async def get_files_modified_map(self, paths: List[str]) -> Dict[str, str]:
        """Return {path: modified_at} for every path that already exists in the DB."""
        result: Dict[str, str] = {}
        conn = self._get_conn()
        batch_size = 900
        for i in range(0, len(paths), batch_size):
            batch = paths[i : i + batch_size]
            placeholders = ",".join("?" for _ in batch)
            query = f"SELECT path, modified_at FROM files WHERE path IN ({placeholders})"
            async with conn.execute(query, batch) as cursor:
                async for row in cursor:
                    result[row[0]] = row[1]
        return result

    async def increment_usage_count(self, file_path: str) -> None:
        """Increments the usage_count for a given file path."""
        conn = self._get_conn()
        await conn.execute(
            "UPDATE files SET usage_count = usage_count + 1 WHERE path = ?",
            (file_path,),
        )
        await conn.commit()

    async def batch_increment_usage(self, file_paths: List[str]) -> None:
        """Increment usage_count for multiple file paths in a single transaction."""
        if not file_paths:
            return
        conn = self._get_conn()
        for path in file_paths:
            await conn.execute(
                "UPDATE files SET usage_count = usage_count + 1 WHERE path = ?",
                (path,),
            )
        await conn.commit()

    async def get_all_files(self) -> List[aiosqlite.Row]:
        """Returns all indexed files ordered by folder and path."""
        conn = self._get_conn()
        async with conn.execute(
            "SELECT path, size, type, folder_tag, usage_count "
            "FROM files ORDER BY folder_tag, path"
        ) as cursor:
            return list(await cursor.fetchall())

    async def get_counts(self) -> Tuple[int, int]:
        """Return (file_count, chunk_count) in a single public call."""
        conn = self._get_conn()
        file_count = 0
        chunk_count = 0
        async with conn.execute("SELECT COUNT(*) FROM files") as cursor:
            row = await cursor.fetchone()
            if row:
                file_count = row[0]
        async with conn.execute("SELECT COUNT(*) FROM chunks") as cursor:
            row = await cursor.fetchone()
            if row:
                chunk_count = row[0]
        return file_count, chunk_count

    async def execute_query(self, sql: str, params: tuple = ()) -> List[Any]:
        """Execute a read-only SQL query and return all rows."""
        conn = self._get_conn()
        async with conn.execute(sql, params) as cursor:
            return list(await cursor.fetchall())

    async def save_query(
        self, question: str, answer: str, source_count: int, latency_ms: float
    ) -> int:
        """Save a query to the history table and return its id."""
        conn = self._get_conn()
        async with conn.execute(
            "INSERT INTO query_history (question, answer, source_count, latency_ms) "
            "VALUES (?, ?, ?, ?) RETURNING id",
            (question, answer, source_count, latency_ms),
        ) as cursor:
            row = await cursor.fetchone()
            await conn.commit()
            return row[0] if row else 0

    async def get_query_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent queries from history."""
        conn = self._get_conn()
        async with conn.execute(
            "SELECT id, question, answer, source_count, latency_ms, created_at "
            "FROM query_history ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "question": r[1],
                    "answer": r[2],
                    "source_count": r[3],
                    "latency_ms": r[4],
                    "created_at": r[5],
                }
                for r in rows
            ]

    async def cleanup_stale_files(self) -> List[str]:
        """Remove index entries for files that no longer exist on disk.

        Returns list of paths that were cleaned up.
        """
        import os

        conn = self._get_conn()
        cleaned: List[str] = []
        async with conn.execute("SELECT id, path FROM files") as cursor:
            rows = list(await cursor.fetchall())
        for row in rows:
            file_id, path = row[0], row[1]
            if not os.path.exists(path):
                await conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
                cleaned.append(path)
                logger.info("Cleaned stale file: %s", path)
        if cleaned:
            await conn.commit()
        return cleaned

    async def clear_all(self) -> Dict[str, int]:
        """Delete ALL indexed data: files, chunks, FTS, and query history.

        Returns counts of removed files and chunks.
        """
        conn = self._get_conn()

        cur = await conn.execute("SELECT COUNT(*) FROM files")
        row = await cur.fetchone()
        files_count = row[0] if row else 0
        await cur.close()

        cur = await conn.execute("SELECT COUNT(*) FROM chunks")
        row = await cur.fetchone()
        chunks_count = row[0] if row else 0
        await cur.close()

        await conn.executescript("""
            -- Remove triggers so chunk deletes don't touch FTS
            DROP TRIGGER IF EXISTS chunks_ai;
            DROP TRIGGER IF EXISTS chunks_ad;
            DROP TRIGGER IF EXISTS chunks_au;

            -- Drop the FTS virtual table entirely
            DROP TABLE IF EXISTS chunk_fts;

            -- Now safe to delete all data
            DELETE FROM chunks;
            DELETE FROM files;
            DELETE FROM query_history;

            -- Recreate FTS table and triggers
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                chunks_text,
                content=chunks,
                content_rowid=id
            );
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
        """)

        logger.info("Cleared all data: %d files, %d chunks", files_count, chunks_count)
        return {"files_removed": files_count, "chunks_removed": chunks_count}

    async def get_files_by_filter(
        self,
        file_type: Optional[str] = None,
        folder_tag: Optional[str] = None,
    ) -> List[aiosqlite.Row]:
        """Return files matching optional type/folder filters."""
        conn = self._get_conn()
        conditions: List[str] = []
        params: List[Any] = []
        if file_type:
            conditions.append("type = ?")
            params.append(file_type)
        if folder_tag:
            conditions.append("folder_tag = ?")
            params.append(folder_tag)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"SELECT path, size, type, folder_tag, usage_count FROM files{where} ORDER BY path"
        async with conn.execute(sql, params) as cursor:
            return list(await cursor.fetchall())

    async def delete_files_by_folder_prefix(self, folder: str) -> None:
        """Delete all files (and cascading chunks) whose path starts with *folder*."""
        conn = self._get_conn()
        await conn.execute(
            "DELETE FROM files WHERE path LIKE ? || '%'",
            (folder,),
        )
        await conn.commit()

    async def is_healthy(self) -> bool:
        """Quick DB health check – runs a trivial query."""
        try:
            conn = self._get_conn()
            async with conn.execute("SELECT 1") as cursor:
                row = await cursor.fetchone()
                return row is not None and row[0] == 1
        except Exception:
            return False

    async def get_all_summaries(self) -> List[Dict[str, Any]]:
        """Return (id, path, summary) for every file that has a non-empty summary."""
        conn = self._get_conn()
        async with conn.execute(
            "SELECT id, path, summary FROM files WHERE summary != '' ORDER BY id"
        ) as cursor:
            rows = await cursor.fetchall()
            return [{"id": r[0], "path": r[1], "summary": r[2]} for r in rows]
