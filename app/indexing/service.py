import os
import logging
import asyncio
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from app.storage.db import DatabaseManager
from app.embeddings.service import EmbeddingService
from app.vector_store.chroma_client import ChromaClient
from app.scanner.scanner import scan_folder as fast_scan
from app.config import settings

logger = logging.getLogger(__name__)


def _resolve_folder_overlaps(folders: List[str]) -> List[Path]:
    """Remove child folders whose content is already covered by a parent.

    Given ["C:/A", "C:/A/B", "D:/X"], returns [Path("C:/A"), Path("D:/X")]
    because scanning C:/A already recurses into C:/A/B.
    """
    resolved: List[Path] = []
    for raw in folders:
        clean = raw.strip().strip('"').strip("'")
        p = Path(clean).resolve()
        if not p.exists() or not p.is_dir():
            continue
        resolved.append(p)

    # Sort by depth (shortest first) so parents come before children
    resolved.sort(key=lambda p: len(p.parts))

    kept: List[Path] = []
    for candidate in resolved:
        # Check if any already-kept folder is a parent of this candidate
        dominated = False
        for parent in kept:
            try:
                candidate.relative_to(parent)
                dominated = True
                logger.info(
                    "Folder overlap detected: '%s' is inside already-queued '%s' — skipping.",
                    candidate, parent,
                )
                break
            except ValueError:
                pass
        if not dominated:
            kept.append(candidate)
    return kept


class IndexingProgress:
    """Thread-safe progress tracker for the indexing pipeline."""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_files = 0
        self.processed_files = 0
        self.total_chunks = 0
        self.skipped_files = 0
        self.new_files = 0
        self.changed_files = 0
        self.status = "idle"
        self.scan_method = ""
        self.scan_duration_ms = 0.0
        self.current_file = ""

    def reset(self, total_files: int):
        with self._lock:
            self.total_files = total_files
            self.processed_files = 0
            self.total_chunks = 0
            self.skipped_files = 0
            self.new_files = 0
            self.changed_files = 0
            self.status = "running"
            self.scan_method = ""
            self.scan_duration_ms = 0.0
            self.current_file = ""

    def update(self, chunks_added: int, current_file: str = ""):
        with self._lock:
            self.processed_files += 1
            self.total_chunks += chunks_added
            self.current_file = current_file

    def complete(self):
        with self._lock:
            self.status = "idle"
            self.current_file = ""

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
        self.supported_extensions = settings.extensions_set
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.max_file_size = settings.max_file_size_bytes
        self._concurrency = settings.index_concurrency

    async def index_folders(self, folders: List[str]):
        """Recursively scans and indexes the provided folders.

        Optimisation pipeline:
        1. **Folder overlap removal** – child folders already covered by a
           parent in the request are dropped so the scanner never visits them
           twice.
        2. **Batch change detection** – after scanning, every discovered file
           path is checked against the DB in a single query.  Files whose
           ``modified_at`` timestamp matches the stored value are skipped
           entirely (no stat, no text extraction, no embedding).
        3. Only *new* and *changed* files proceed to the full index pipeline.
        """
        if indexing_lock.locked():
            logger.warning("Indexing is already in progress. Skipping duplicate request.")
            return

        async with indexing_lock:
            # ── 1. Resolve folder overlaps ───────────────────────────────
            unique_folders = _resolve_folder_overlaps(folders)
            if not unique_folders:
                logger.warning("No valid folders to index after overlap resolution.")
                progress.status = "idle"
                return

            logger.info(
                "Starting indexing for %d folder(s) (after overlap pruning): %s",
                len(unique_folders),
                [str(f) for f in unique_folders],
            )

            # ── 2. Scan for candidate files ──────────────────────────────
            all_files: List[Tuple[Path, str]] = []
            seen_paths: Set[str] = set()  # dedup across scan results
            scan_method = ""
            scan_duration = 0.0

            for path in unique_folders:
                scan_result = fast_scan(path, self.supported_extensions)
                scan_method = scan_result.method
                scan_duration += scan_result.duration_ms

                for file_path in scan_result.files:
                    abs_key = str(file_path.resolve())
                    if abs_key not in seen_paths:
                        seen_paths.add(abs_key)
                        all_files.append((file_path, path.name))

            if not all_files:
                logger.info(
                    "No files with extensions %s found in %s.",
                    self.supported_extensions,
                    [str(f) for f in unique_folders],
                )
                progress.status = "idle"
                return

            # ── 3. Batch change detection ────────────────────────────────
            file_paths_list = [str(fp.absolute()) for fp, _ in all_files]
            indexed_map = await self.db.get_files_modified_map(file_paths_list)

            files_to_index: List[Tuple[Path, str]] = []
            skipped = 0
            new_count = 0
            changed_count = 0

            for file_path, folder_tag in all_files:
                abs_path = str(file_path.absolute())
                try:
                    current_mtime = datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    ).isoformat()
                except OSError:
                    # File vanished between scan and check — skip
                    skipped += 1
                    continue

                stored_mtime = indexed_map.get(abs_path)
                if stored_mtime is not None and stored_mtime == current_mtime:
                    # Unchanged — skip entirely
                    skipped += 1
                    continue

                if stored_mtime is None:
                    new_count += 1
                else:
                    changed_count += 1
                files_to_index.append((file_path, folder_tag))

            logger.info(
                "Change detection: %d scanned → %d to index "
                "(%d new, %d changed, %d unchanged/skipped).",
                len(all_files),
                len(files_to_index),
                new_count,
                changed_count,
                skipped,
            )

            # Progress tracks only files that actually need work
            progress.reset(len(files_to_index))
            progress.scan_method = scan_method
            progress.scan_duration_ms = scan_duration
            progress.skipped_files = skipped
            progress.new_files = new_count
            progress.changed_files = changed_count

            if not files_to_index:
                logger.info("All files are up-to-date — nothing to index.")
                progress.complete()
                return

            # ── 4. Index only new / changed files (with bounded concurrency) ──
            sem = asyncio.Semaphore(self._concurrency)

            async def _index_with_sem(fp: Path, ft: str):
                async with sem:
                    await self.index_file(fp, ft)

            tasks = [
                _index_with_sem(file_path, folder_tag)
                for file_path, folder_tag in files_to_index
            ]
            await asyncio.gather(*tasks)

            progress.complete()
            logger.info(
                "Indexing completed: %d files processed (%d skipped).",
                len(files_to_index),
                skipped,
            )

    async def index_file(self, path: Path, folder_tag: str):
        """Extracts text, chunks it, generates embeddings, and stores in both DBs."""
        chunks_added = 0
        try:
            stat = path.stat()

            if stat.st_size > self.max_file_size:
                logger.warning(
                    "Skipping oversized file (%d MB): %s",
                    stat.st_size // (1024 * 1024), path,
                )
                progress.update(0, str(path.name))
                return

            file_data = {
                "path": str(path.absolute()),
                "size": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "type": path.suffix.lower(),
                "folder_tag": folder_tag
            }

            # If file existed before (changed), cleanup old chunks
            existing_file = await self.db.get_file_by_path(file_data["path"])
            if existing_file:
                file_id = existing_file["id"]
                old_chunks = await self.db.get_file_chunks(file_id)
                old_chunk_ids = [str(c["id"]) for c in old_chunks]
                await self.chroma_client.delete_documents(old_chunk_ids)
                await self.db.delete_file_chunks(file_id)

            # Extract text (run in executor to avoid blocking the event loop)
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, self._extract_text, path)
            if not text:
                progress.update(0, str(path.name))
                return

            # Generate extractive summary
            summary = self._generate_summary(text, path)

            # Insert/Update file metadata
            file_data["summary"] = summary
            file_id = await self.db.insert_file(file_data)

            # Create chunks
            chunks_data = self._create_chunks(text, file_path=str(path))

            # Bulk insert chunks
            for chunk in chunks_data:
                chunk["file_id"] = file_id
            chunk_ids_int = await self.db.insert_chunks_bulk(chunks_data)
            chunk_ids = [str(cid) for cid in chunk_ids_int]
            chunk_texts = [c["text_preview"] for c in chunks_data]

            # Commit all chunk inserts in one batch
            await self.db.commit()

            # Batch add to Chroma
            if chunk_texts:
                embeddings = await self.embedding_service.embed_texts(chunk_texts)
                metadatas = [
                    {"chunk_id": cid, "file_path": str(path), "folder_tag": folder_tag} 
                    for cid in chunk_ids
                ]
                await self.chroma_client.add_documents(chunk_ids, embeddings, metadatas)
                chunks_added = len(chunk_ids)

            logger.info("Indexed file: %s (%d chunks)", path, chunks_added)

            # Store document-level summary embedding in the summaries collection
            if summary:
                try:
                    summary_emb = await self.embedding_service.embed_texts([summary])
                    await self.chroma_client.add_summary(
                        doc_id=f"file_{file_id}",
                        embedding=summary_emb[0],
                        metadata={
                            "file_id": file_id,
                            "file_path": str(path),
                            "folder_tag": folder_tag,
                        },
                    )
                except Exception as e:
                    logger.warning("Failed to store summary embedding for %s: %s", path, e)

        except Exception as e:
            logger.error("Error indexing file %s: %s", path, e)
        finally:
            progress.update(chunks_added, str(path.name))

    def _extract_text(self, path: Path) -> str:
        """Text extraction for multiple file types.

        Supports: .txt, .md, .pdf, .docx, .csv, .json, and source code files
        (.py, .js, .ts, .java, .c, .cpp, .rs, .go, .rb, .html, .css, .xml,
        .yaml, .yml, .toml, .ini, .cfg, .sh, .bat).
        """
        ext = path.suffix.lower()

        # Plain text / Markdown / source code
        text_extensions = {
            ".txt", ".md", ".py", ".js", ".ts", ".java", ".c", ".cpp", ".rs",
            ".go", ".rb", ".html", ".css", ".xml", ".yaml", ".yml", ".toml",
            ".ini", ".cfg", ".sh", ".bat",
        }
        if ext in text_extensions:
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.error("Error reading text file %s: %s", path, e)
                return ""

        # PDF
        if ext == ".pdf":
            try:
                import pdfplumber
                text_content: list[str] = []
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
                logger.error("Error reading PDF file %s: %s", path, e)
                return ""

        # DOCX
        if ext == ".docx":
            try:
                from docx import Document
                doc = Document(str(path))
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                return "\n".join(paragraphs)
            except ImportError:
                logger.error("python-docx not installed. Cannot index DOCX files.")
                return ""
            except Exception as e:
                logger.error("Error reading DOCX file %s: %s", path, e)
                return ""

        # CSV
        if ext == ".csv":
            try:
                import csv
                rows: list[str] = []
                with open(path, encoding="utf-8", errors="replace", newline="") as f:
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        if i > 5000:
                            break  # limit rows to avoid OOM
                        rows.append(", ".join(row))
                return "\n".join(rows)
            except Exception as e:
                logger.error("Error reading CSV file %s: %s", path, e)
                return ""

        # JSON
        if ext == ".json":
            try:
                import json
                text = path.read_text(encoding="utf-8", errors="replace")
                # Pretty-print for better chunking
                data = json.loads(text)
                return json.dumps(data, indent=2, ensure_ascii=False)[:200_000]
            except Exception as e:
                logger.error("Error reading JSON file %s: %s", path, e)
                return ""

        return ""

    def _generate_summary(self, text: str, path: Path, max_chars: int = 300) -> str:
        """Build a short extractive summary for document-level indexing.

        Format: ``[TYPE: filename] first N characters (sentence-snapped)``
        This is used as the document-level embedding in the summary
        collection so queries can first identify *which* files are relevant
        before drilling into individual chunks.
        """
        fname = path.name
        ftype = path.suffix.lstrip(".").upper() or "file"
        prefix = f"[{ftype}: {fname}] "

        # Take the beginning of the document
        raw = text[:max_chars + 80]  # overshoot to allow sentence snap
        # Snap to sentence end within the max_chars window
        boundary = self._find_sentence_boundary(raw, min(max_chars, len(raw)))
        if boundary <= 0:
            boundary = min(max_chars, len(raw))
        snippet = raw[:boundary].strip()
        return prefix + snippet

    @staticmethod
    def _find_sentence_boundary(text: str, pos: int, window: int = 80) -> int:
        """Find the nearest sentence-ending punctuation near *pos*.

        Looks backwards up to *window* chars for '.', '!', '?', or '\n\n'.
        Returns the index **after** the delimiter so the next chunk starts
        on a clean sentence.  Falls back to *pos* if nothing is found.
        """
        search_start = max(0, pos - window)
        region = text[search_start:pos]
        # Prefer paragraph break, then sentence-ending punctuation
        for delim in ["\n\n", ". ", "! ", "? ", ".\n", "!\n", "?\n"]:
            idx = region.rfind(delim)
            if idx != -1:
                return search_start + idx + len(delim)
        return pos

    def _create_chunks(self, text: str, file_path: str = "") -> List[Dict[str, Any]]:
        """Splits text into sentence-aware, overlapping chunks.

        For Markdown files, first splits on headings before falling back
        to character-based chunking.  A file-context prefix is prepended
        to every chunk so the embedding captures document-level context.
        """
        chunks: List[Dict[str, Any]] = []
        if not text:
            return chunks

        # Build a short context prefix from the file path
        context_prefix = ""
        if file_path:
            fname = Path(file_path).name
            ftype = Path(file_path).suffix.lstrip(".").upper() or "file"
            context_prefix = f"[{ftype}: {fname}] "

        # Markdown-aware splitting: split on heading boundaries first
        ext = Path(file_path).suffix.lower() if file_path else ""
        if ext == ".md":
            import re
            sections = re.split(r'(?=^#{1,3}\s)', text, flags=re.MULTILINE)
            sections = [s for s in sections if s.strip()]
            for section in sections:
                if len(section) <= self.chunk_size:
                    chunks.append({
                        "start_offset": text.find(section),
                        "end_offset": text.find(section) + len(section),
                        "text_preview": context_prefix + section.strip(),
                    })
                else:
                    # Sub-chunk large sections
                    sub_chunks = self._split_text(section, context_prefix, text.find(section))
                    chunks.extend(sub_chunks)
            if chunks:
                return chunks

        # Default: character-based chunking with sentence snapping
        chunks = self._split_text(text, context_prefix, 0)
        return chunks

    def _split_text(
        self, text: str, context_prefix: str, base_offset: int
    ) -> List[Dict[str, Any]]:
        """Character-based chunking with sentence-boundary snapping."""
        chunks: List[Dict[str, Any]] = []
        start = 0
        text_len = len(text)
        while start < text_len:
            raw_end = min(start + self.chunk_size, text_len)
            end = (
                self._find_sentence_boundary(text, raw_end)
                if raw_end < text_len
                else raw_end
            )
            if end <= start:
                end = raw_end

            chunk_text = context_prefix + text[start:end]
            chunks.append({
                "start_offset": base_offset + start,
                "end_offset": base_offset + end,
                "text_preview": chunk_text,
            })
            start = end - self.chunk_overlap if end < text_len else text_len
        return chunks
