import os
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.storage.db import DatabaseManager
from app.embeddings.service import EmbeddingService
from app.vector_store.chroma_client import ChromaClient

logger = logging.getLogger(__name__)

class IndexingProgress:
    def __init__(self):
        self.total_files = 0
        self.processed_files = 0
        self.total_chunks = 0
        self.status = "idle"

    def reset(self, total_files: int):
        self.total_files = total_files
        self.processed_files = 0
        self.total_chunks = 0
        self.status = "running"

    def update(self, chunks_added: int):
        self.processed_files += 1
        self.total_chunks += chunks_added

    def complete(self):
        self.status = "idle"

# Global progress tracker and lock
progress = IndexingProgress()
indexing_lock = asyncio.Lock()

class IndexingService:
    def __init__(
        self, 
        db: DatabaseManager, 
        embedding_service: EmbeddingService, 
        chroma_client: ChromaClient
    ):
        self.db = db
        self.embedding_service = embedding_service
        self.chroma_client = chroma_client
        self.supported_extensions = {".txt", ".md", ".pdf"}
        self.chunk_size = 400
        self.chunk_overlap = 100

    async def index_folders(self, folders: List[str]):
        """Recursively scans and indexes the provided folders."""
        if indexing_lock.locked():
            logger.warning("Indexing is already in progress. Skipping duplicate request.")
            return

        async with indexing_lock:
            all_files = []
            logger.info(f"Starting indexing for folders: {folders}")
            for folder_path in folders:
                # Sanitize path (strip quotes if users paste them)
                clean_path = folder_path.strip().strip('"').strip("'")
                path = Path(clean_path)
                
                if not path.exists():
                    logger.warning(f"Folder does not exist: {clean_path}")
                    continue
                
                if not path.is_dir():
                    logger.warning(f"Path is not a directory: {clean_path}")
                    continue

                for file_path in path.rglob("*"):
                    if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                        # Capture folder_tag relative to the root
                        all_files.append((file_path, path.name))
            
            if not all_files:
                logger.info(f"No files with extensions {self.supported_extensions} found in {folders}.")
                progress.status = "idle"
                return

            progress.reset(len(all_files))
            logger.info(f"Found {len(all_files)} files to index.")
            
            # Process files
            for file_path, folder_tag in all_files:
                await self.index_file(file_path, folder_tag)
                
            progress.complete()
            logger.info(f"Indexing completed for {len(all_files)} files.")

    async def index_file(self, path: Path, folder_tag: str):
        """Extracts text, chunks it, generates embeddings, and stores in both DBs."""
        chunks_added = 0
        try:
            stat = path.stat()
            file_data = {
                "path": str(path.absolute()),
                "size": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "type": path.suffix.lower(),
                "folder_tag": folder_tag
            }

            # Check if file has changed since last indexing
            existing_file = await self.db.get_file_by_path(file_data["path"])
            if existing_file:
                if existing_file["modified_at"] == file_data["modified_at"]:
                    # Even if skipped, we mark as processed for progress bar
                    progress.update(0) 
                    return
                # If modified, cleanup old
                file_id = existing_file["id"]
                old_chunks = await self.db.get_file_chunks(file_id)
                old_chunk_ids = [str(c["id"]) for c in old_chunks]
                await self.chroma_client.delete_documents(old_chunk_ids)
                await self.db.delete_file_chunks(file_id)

            # Extract text
            text = await self._extract_text(path)
            if not text:
                progress.update(0)
                return

            # Insert/Update file metadata
            file_id = await self.db.insert_file(file_data)

            # Create chunks
            chunks_data = self._create_chunks(text)
            chunk_texts = []
            chunk_ids = []
            
            for chunk in chunks_data:
                chunk["file_id"] = file_id
                chunk_id = await self.db.insert_chunk(chunk)
                chunk_ids.append(str(chunk_id))
                chunk_texts.append(chunk["text_preview"])

            # Batch add to Chroma
            if chunk_texts:
                embeddings = await self.embedding_service.embed_texts(chunk_texts)
                metadatas = [
                    {"chunk_id": cid, "file_path": str(path), "folder_tag": folder_tag} 
                    for cid in chunk_ids
                ]
                await self.chroma_client.add_documents(chunk_ids, embeddings, metadatas)
                chunks_added = len(chunk_ids)

            logger.info(f"Indexed file: {path} ({chunks_added} chunks)")

        except Exception as e:
            logger.error(f"Error indexing file {path}: {e}")
        finally:
            progress.update(chunks_added)

    async def _extract_text(self, path: Path) -> str:
        """Text extraction for .txt, .md, and .pdf."""
        ext = path.suffix.lower()
        if ext in {".txt", ".md"}:
            try:
                # Use a slightly more robust way to read local text files
                content = path.read_text(encoding="utf-8", errors="replace")
                return content
            except Exception as e:
                logger.error(f"Error reading text file {path}: {e}")
                return ""
        elif ext == ".pdf":
            try:
                import pdfplumber
                text_content = []
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text)
                return "\n".join(text_content)
            except ImportError:
                logger.error("pdfplumber not installed. Cannot index PDF files.")
                return ""
            except Exception as e:
                logger.error(f"Error reading PDF file {path}: {e}")
                return ""
        return ""

    def _create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Splits text into overlapping chunks."""
        chunks = []
        if not text:
            return chunks

        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            chunks.append({
                "start_offset": start,
                "end_offset": min(end, len(text)),
                "text_preview": chunk_text
            })
            
            start += (self.chunk_size - self.chunk_overlap)
            if start >= len(text):
                break
                
        return chunks
