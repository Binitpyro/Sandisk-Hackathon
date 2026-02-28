import logging
from typing import Dict, Any, List
from app.storage.db import DatabaseManager

logger = logging.getLogger(__name__)

class InsightsService:
    def __init__(self, db: DatabaseManager):
        self.db = db

    async def get_stats(self) -> Dict[str, Any]:
        """Aggregates storage insights from the database."""
        stats = {
            "total_size_bytes": 0,
            "file_count": 0,
            "top_files": [],
            "cold_files": [],
            "type_breakdown": {}
        }

        try:
            # 1. Basic Stats
            async with self.db.conn.execute("SELECT SUM(size), COUNT(*) FROM files") as cursor:
                row = await cursor.fetchone()
                stats["total_size_bytes"] = row[0] or 0
                stats["file_count"] = row[1] or 0

            # 2. Top N Largest Files
            async with self.db.conn.execute(
                "SELECT path, size FROM files ORDER BY size DESC LIMIT 10"
            ) as cursor:
                rows = await cursor.fetchall()
                stats["top_files"] = [{"path": r[0], "size": r[1]} for r in rows]

            # 3. Cold Files (Usage Count = 0, ordered by size)
            async with self.db.conn.execute(
                "SELECT path, size FROM files WHERE usage_count = 0 ORDER BY size DESC LIMIT 10"
            ) as cursor:
                rows = await cursor.fetchall()
                stats["cold_files"] = [{"path": r[0], "size": r[1]} for r in rows]

            # 4. Type Breakdown
            async with self.db.conn.execute(
                "SELECT type, COUNT(*), SUM(size) FROM files GROUP BY type"
            ) as cursor:
                rows = await cursor.fetchall()
                stats["type_breakdown"] = {
                    r[0]: {"count": r[1], "size": r[2]} for r in rows
                }

        except Exception as e:
            logger.error(f"Error fetching insights: {e}")

        return stats
