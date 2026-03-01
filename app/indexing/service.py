import os
import logging
import asyncio
import threading
import concurrent.futures
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

    resolved.sort(key=lambda p: len(p.parts))

    kept: List[Path] = []
    for candidate in resolved:
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

            # Set status to 'running' before scan so SSE stream stays open
            progress.reset(0)
            progress.status = "running"
            progress.current_file = "Scanning folders…"

            loop = asyncio.get_running_loop()
            all_files, scan_method, scan_duration = await loop.run_in_executor(
                None, self._scan_all_folders, unique_folders
            )

            if not all_files:
                logger.info(
                    "No files with extensions %s found in %s.",
                    self.supported_extensions,
                    [str(f) for f in unique_folders],
                )
                progress.status = "idle"
                return

            files_to_index, skipped, new_count, changed_count = await self._detect_changes(
                all_files
            )

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

            await self._batch_index_pipeline(files_to_index)

            progress.complete()
            logger.info(
                "Indexing completed: %d files processed (%d skipped).",
                len(files_to_index),
                skipped,
            )

    # ------------------------------------------------------------------ #
    #  Three-phase batch pipeline (extract → embed → store)              #
    # ------------------------------------------------------------------ #

    async def _batch_index_pipeline(
        self, files_to_index: List[Tuple[Path, str]]
    ) -> None:
        """Optimised pipeline that batches work across *all* files.

        Phase 1 – **Parallel text extraction + chunking** (I/O-bound,
                  runs in a ThreadPoolExecutor so PDF/DOCX parsing
                  doesn't block the event loop).
        Phase 2 – **Batch embedding** of every chunk + summary in one
                  call to ``SentenceTransformer.encode``.  This is the
                  single biggest speed-up: the model processes a large
                  matrix in one GPU/CPU pass instead of many small ones.
        Phase 3 – **Batch storage** into SQLite + ChromaDB.
        """
        total = len(files_to_index)

        # ── Phase 1: parallel extract + chunk ──────────────────────────
        logger.info("Pipeline phase 1/3: extracting text from %d files …", total)
        max_workers = min(self._concurrency * 2, (os.cpu_count() or 4) + 2)
        loop = asyncio.get_running_loop()

        prepared: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [
                loop.run_in_executor(pool, self._extract_and_prepare, fp, ft)
                for fp, ft in files_to_index
            ]
            results = await asyncio.gather(*futs, return_exceptions=True)

        for (fp, _ft), result in zip(files_to_index, results):
            if isinstance(result, Exception):
                logger.error("Error extracting %s: %s", fp, result)
                progress.update(0, fp.name)
                continue
            if result is None:
                progress.update(0, fp.name)
                continue
            prepared.append(result)

        if not prepared:
            return

        # ── Phase 2: batch embed ALL texts at once ─────────────────────
        all_texts: List[str] = []
        text_map: List[Tuple[int, str, int]] = []  # (file_idx, kind, sub_idx)

        for fi, item in enumerate(prepared):
            for ci, chunk in enumerate(item["chunks"]):
                all_texts.append(chunk["text_preview"])
                text_map.append((fi, "chunk", ci))
            if item.get("summary"):
                all_texts.append(item["summary"])
                text_map.append((fi, "summary", 0))

        logger.info(
            "Pipeline phase 2/3: embedding %d texts in batch …", len(all_texts)
        )
        all_embeddings = await self.embedding_service.embed_texts(
            all_texts, batch_size=settings.embedding_batch_size
        )

        # Distribute embeddings back to their owning file
        for idx, (fi, kind, si) in enumerate(text_map):
            if kind == "chunk":
                prepared[fi]["chunks"][si]["_embedding"] = all_embeddings[idx]
            else:
                prepared[fi]["_summary_embedding"] = all_embeddings[idx]

        # ── Phase 3: batch store in DB + ChromaDB ──────────────────────
        logger.info("Pipeline phase 3/3: storing %d files …", len(prepared))

        all_chroma_ids: List[str] = []
        all_chroma_embs: List[List[float]] = []
        all_chroma_metas: List[Dict[str, Any]] = []
        summary_items: List[Dict[str, Any]] = []

        for item in prepared:
            try:
                fdata = item["file_data"]
                fpath_str = fdata["path"]
                ftag = item["folder_tag"]

                # If file already exists, remove its old chunks
                existing = await self.db.get_file_by_path(fpath_str)
                if existing:
                    fid_old = existing["id"]
                    old_chunks = await self.db.get_file_chunks(fid_old)
                    old_ids = [str(c["id"]) for c in old_chunks]
                    if old_ids:
                        await self.chroma_client.delete_documents(old_ids)
                    await self.db.delete_file_chunks(fid_old)

                file_id = await self.db.insert_file(fdata)

                # Prepare chunk rows (strip the transient _embedding key)
                chunk_rows = [
                    {k: v for k, v in c.items() if k != "_embedding"}
                    for c in item["chunks"]
                ]
                for cr in chunk_rows:
                    cr["file_id"] = file_id

                chunk_ids_int = await self.db.insert_chunks_bulk(chunk_rows)

                for cid_int, chunk in zip(chunk_ids_int, item["chunks"]):
                    all_chroma_ids.append(str(cid_int))
                    all_chroma_embs.append(chunk["_embedding"])
                    all_chroma_metas.append({
                        "chunk_id": str(cid_int),
                        "file_path": fpath_str,
                        "folder_tag": ftag,
                    })

                if item.get("_summary_embedding"):
                    summary_items.append({
                        "doc_id": f"file_{file_id}",
                        "embedding": item["_summary_embedding"],
                        "metadata": {
                            "file_id": file_id,
                            "file_path": fpath_str,
                            "folder_tag": ftag,
                        },
                    })

                progress.update(len(item["chunks"]), item["path"].name)

            except Exception as e:
                logger.error("Error storing file %s: %s", item["path"], e)
                progress.update(0, item["path"].name)

        await self.db.commit()

        # Batch upsert into ChromaDB (ChromaDB caps at ~5 461 per call)
        CHROMA_BATCH = 5000
        for i in range(0, len(all_chroma_ids), CHROMA_BATCH):
            end = min(i + CHROMA_BATCH, len(all_chroma_ids))
            await self.chroma_client.add_documents(
                all_chroma_ids[i:end],
                all_chroma_embs[i:end],
                all_chroma_metas[i:end],
            )

        # Batch summary upserts
        if summary_items:
            await self.chroma_client.add_summaries_batch(summary_items)

    def _extract_and_prepare(
        self, path: Path, folder_tag: str
    ) -> Optional[Dict[str, Any]]:
        """Synchronous helper: extract text → chunk → build metadata.

        Designed to run inside a ThreadPoolExecutor.
        """
        try:
            stat = path.stat()
            if stat.st_size > self.max_file_size:
                logger.warning("Skipping oversized file (%d MB): %s",
                               stat.st_size // (1024 * 1024), path)
                return None

            text = self._extract_text(path)
            if not text:
                return None

            summary = self._generate_summary(text, path)
            chunks = self._create_chunks(text, file_path=str(path))
            if not chunks:
                return None

            file_data = {
                "path": str(path.absolute()),
                "size": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "type": path.suffix.lower(),
                "folder_tag": folder_tag,
                "summary": summary,
            }

            return {
                "path": path,
                "folder_tag": folder_tag,
                "file_data": file_data,
                "chunks": chunks,
                "summary": summary,
            }
        except Exception as e:
            logger.error("Error preparing %s: %s", path, e)
            return None

    def _scan_all_folders(
        self, unique_folders: List[Path]
    ) -> Tuple[List[Tuple[Path, str]], str, float]:
        """Scan all folders and return deduplicated files with scan metadata."""
        all_files: List[Tuple[Path, str]] = []
        seen_paths: Set[str] = set()
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

        return all_files, scan_method, scan_duration

    async def _detect_changes(
        self, all_files: List[Tuple[Path, str]]
    ) -> Tuple[List[Tuple[Path, str]], int, int, int]:
        """Compare scanned files against the DB and classify as new/changed/unchanged.

        Returns (files_to_index, skipped_count, new_count, changed_count).
        """
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
                skipped += 1
                continue

            stored_mtime = indexed_map.get(abs_path)
            if stored_mtime is not None and stored_mtime == current_mtime:
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
        return files_to_index, skipped, new_count, changed_count

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

            existing_file = await self.db.get_file_by_path(file_data["path"])
            if existing_file:
                file_id = existing_file["id"]
                old_chunks = await self.db.get_file_chunks(file_id)
                old_chunk_ids = [str(c["id"]) for c in old_chunks]
                await self.chroma_client.delete_documents(old_chunk_ids)
                await self.db.delete_file_chunks(file_id)

            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, self._extract_text, path)
            if not text:
                progress.update(0, str(path.name))
                return

            summary = self._generate_summary(text, path)

            file_data["summary"] = summary
            file_id = await self.db.insert_file(file_data)

            chunks_data = self._create_chunks(text, file_path=str(path))

            for chunk in chunks_data:
                chunk["file_id"] = file_id
            chunk_ids_int = await self.db.insert_chunks_bulk(chunks_data)
            chunk_ids = [str(cid) for cid in chunk_ids_int]
            chunk_texts = [c["text_preview"] for c in chunks_data]

            await self.db.commit()

            if chunk_texts:
                embeddings = await self.embedding_service.embed_texts(chunk_texts)
                metadatas = [
                    {"chunk_id": cid, "file_path": str(path), "folder_tag": folder_tag} 
                    for cid in chunk_ids
                ]
                await self.chroma_client.add_documents(chunk_ids, embeddings, metadatas)
                chunks_added = len(chunk_ids)

            logger.info("Indexed file: %s (%d chunks)", path, chunks_added)

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

    _TEXT_EXTENSIONS = frozenset({
        ".txt", ".md", ".py", ".js", ".ts", ".java", ".c", ".cpp", ".rs",
        ".go", ".rb", ".html", ".css", ".xml", ".yaml", ".yml", ".toml",
        ".ini", ".cfg", ".sh", ".bat",
    })

    def _extract_text(self, path: Path) -> str:
        """Text extraction for multiple file types.

        Supports: .txt, .md, .pdf, .docx, .csv, .json, and source code files.
        """
        ext = path.suffix.lower()
        extractor = {
            ".pdf": self._extract_pdf,
            ".docx": self._extract_docx,
            ".csv": self._extract_csv,
            ".json": self._extract_json,
        }.get(ext)

        if extractor:
            return extractor(path)

        if ext in self._TEXT_EXTENSIONS:
            return self._extract_plain_text(path)

        return ""

    @staticmethod
    def _extract_plain_text(path: Path) -> str:
        """Read a plain-text or source-code file."""
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error("Error reading text file %s: %s", path, e)
            return ""

    @staticmethod
    def _extract_pdf(path: Path) -> str:
        """Extract text from a PDF file."""
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

    @staticmethod
    def _extract_docx(path: Path) -> str:
        """Extract text from a DOCX file."""
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

    @staticmethod
    def _extract_csv(path: Path) -> str:
        """Extract text from a CSV file (caps at 5000 rows)."""
        try:
            import csv
            rows: list[str] = []
            with open(path, encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i > 5000:
                        break
                    rows.append(", ".join(row))
            return "\n".join(rows)
        except Exception as e:
            logger.error("Error reading CSV file %s: %s", path, e)
            return ""

    @staticmethod
    def _extract_json(path: Path) -> str:
        """Extract text from a JSON file.

        Handles common real-world quirks:
        - UTF-8 BOM (``utf-8-sig``)
        - Trailing commas, unquoted keys, single-quoted strings (via fallback)
        - JSONL / multi-object files (``Extra data`` errors)
        """
        import json
        import re

        try:
            # Try utf-8-sig first to handle BOM transparently
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except Exception as e:
            logger.error("Error reading JSON file %s: %s", path, e)
            return ""

        # --- Attempt 1: strict parse --------------------------------
        try:
            data = json.loads(text)
            return json.dumps(data, indent=2, ensure_ascii=False)[:200_000]
        except json.JSONDecodeError:
            pass

        # --- Attempt 2: strip trailing commas before } and ] --------
        try:
            cleaned = re.sub(r',\s*([}\]])', r'\1', text)
            data = json.loads(cleaned)
            return json.dumps(data, indent=2, ensure_ascii=False)[:200_000]
        except json.JSONDecodeError:
            pass

        # --- Attempt 3: JSONL (one JSON object per line) ------------
        try:
            parts: list[str] = []
            for line in text.splitlines():
                line = line.strip()
                if line:
                    parts.append(
                        json.dumps(json.loads(line), indent=2, ensure_ascii=False)
                    )
            if parts:
                return "\n".join(parts)[:200_000]
        except json.JSONDecodeError:
            pass

        # --- Fallback: treat as plain text so the file isn't skipped -
        logger.debug("JSON parse failed for %s — falling back to raw text.", path)
        return text[:200_000]

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

        raw = text[:max_chars + 80]  # overshoot to allow sentence snap
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
        if not text:
            return []

        context_prefix = self._build_context_prefix(file_path)

        ext = Path(file_path).suffix.lower() if file_path else ""
        if ext == ".md":
            md_chunks = self._chunk_markdown(text, context_prefix)
            if md_chunks:
                return md_chunks

        return self._split_text(text, context_prefix, 0)

    @staticmethod
    def _build_context_prefix(file_path: str) -> str:
        """Build a ``[TYPE: filename]`` prefix for chunk embeddings."""
        if not file_path:
            return ""
        fname = Path(file_path).name
        ftype = Path(file_path).suffix.lstrip(".").upper() or "file"
        return f"[{ftype}: {fname}] "

    def _chunk_markdown(
        self, text: str, context_prefix: str
    ) -> List[Dict[str, Any]]:
        """Split Markdown text by headings into section-aware chunks."""
        import re
        sections = re.split(r'(?=^#{1,3}\s)', text, flags=re.MULTILINE)
        sections = [s for s in sections if s.strip()]

        chunks: List[Dict[str, Any]] = []
        offset = 0
        for section in sections:
            section_start = text.find(section, offset)
            if section_start == -1:
                section_start = offset
            if len(section) <= self.chunk_size:
                chunks.append({
                    "start_offset": section_start,
                    "end_offset": section_start + len(section),
                    "text_preview": context_prefix + section.strip(),
                })
            else:
                chunks.extend(self._split_text(section, context_prefix, section_start))
            offset = section_start + len(section)
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
