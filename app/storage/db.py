import aiosqlite
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "pma_metadata.db"):
        self.db_path = db_path
        self.conn: Optional[aiosqlite.Connection] = None

    async def connect(self):
        if not self.conn:
            self.conn = await aiosqlite.connect(self.db_path)
            self.conn.row_factory = aiosqlite.Row
            # Enable foreign keys
            await self.conn.execute("PRAGMA foreign_keys = ON;")

    async def close(self):
        if self.conn:
            await self.conn.close()
            self.conn = None

    async def init_db(self, schema_path: str = "app/storage/schema.sql"):
        await self.connect()
        try:
            with open(schema_path, "r") as f:
                schema = f.read()
            await self.conn.executescript(schema)
            await self.conn.commit()
            logger.info("Database initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    async def insert_file(self, file_data: Dict[str, Any]) -> int:
        """Inserts file metadata and returns the new file id."""
        query = """
        INSERT INTO files (path, size, modified_at, type, folder_tag)
        VALUES (:path, :size, :modified_at, :type, :folder_tag)
        ON CONFLICT(path) DO UPDATE SET
            size=excluded.size,
            modified_at=excluded.modified_at,
            type=excluded.type,
            folder_tag=excluded.folder_tag
        RETURNING id;
        """
        async with self.conn.execute(query, file_data) as cursor:
            row = await cursor.fetchone()
            file_id = row[0]
            await self.conn.commit()
            return file_id

    async def insert_chunk(self, chunk_data: Dict[str, Any]) -> int:
        """Inserts a chunk and returns the new chunk id."""
        query = """
        INSERT INTO chunks (file_id, start_offset, end_offset, text_preview)
        VALUES (:file_id, :start_offset, :end_offset, :text_preview)
        RETURNING id;
        """
        async with self.conn.execute(query, chunk_data) as cursor:
            row = await cursor.fetchone()
            chunk_id = row[0]
            await self.conn.commit()
            return chunk_id

    async def delete_file_chunks(self, file_id: int):
        """Deletes all chunks associated with a file."""
        await self.conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        await self.conn.commit()

    async def get_file_chunks(self, file_id: int) -> List[aiosqlite.Row]:
        """Returns all chunks for a given file id."""
        async with self.conn.execute("SELECT * FROM chunks WHERE file_id = ?", (file_id,)) as cursor:
            return await cursor.fetchall()
    
    async def get_file_by_path(self, path: str) -> Optional[aiosqlite.Row]:
        """Returns file metadata by path."""
        async with self.conn.execute("SELECT * FROM files WHERE path = ?", (path,)) as cursor:
            return await cursor.fetchone()

    async def increment_usage_count(self, file_path: str):
        """Increments the usage_count for a given file path."""
        await self.conn.execute(
            "UPDATE files SET usage_count = usage_count + 1 WHERE path = ?", 
            (file_path,)
        )
        await self.conn.commit()
