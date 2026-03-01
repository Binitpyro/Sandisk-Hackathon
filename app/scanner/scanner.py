"""
Unified file scanner with automatic strategy selection.

Strategies (tried in order):

1. **NTFS MFT** – reads the Master File Table directly via Windows API.
   Requires admin privileges on an NTFS volume.  Can enumerate millions
   of files in 2-5 seconds (same approach as TreeSize / Everything).

2. **os.scandir walk** – cross-platform fallback that is still faster
   than ``Path.rglob`` because ``DirEntry`` objects cache ``stat()``
   results and we avoid repeated system calls.
"""

import os
import logging
import time
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Outcome of a folder scan."""

    files: List[Path] = field(default_factory=list)
    method: str = "scandir"          # "ntfs_mft" | "scandir"
    duration_ms: float = 0.0
    total_mft_entries: int = 0       # only set for MFT scans


def scan_folder(folder: Path, extensions: Set[str]) -> ScanResult:
    """
    Scan *folder* for files matching *extensions*.

    Automatically picks the fastest available backend.
    """
    t0 = time.perf_counter()

    # ── Strategy 1: NTFS MFT (Windows + admin) ──────────────────────────
    if platform.system() == "Windows":
        try:
            from app.scanner.ntfs_mft import NTFSScanner

            scanner = NTFSScanner()
            paths = scanner.scan_folder(folder, extensions)
            if paths is not None:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                logger.info(
                    "NTFS MFT scan: %d files in %.0f ms "
                    "(%s total MFT entries)",
                    len(paths),
                    elapsed_ms,
                    f"{scanner.entry_count:,}",
                )
                return ScanResult(
                    files=paths,
                    method="ntfs_mft",
                    duration_ms=elapsed_ms,
                    total_mft_entries=scanner.entry_count,
                )
        except (ImportError, OSError, PermissionError) as e:
            logger.debug("NTFS MFT scan unavailable: %s", e)

    # ── Strategy 2: os.scandir recursive walk ────────────────────────────
    files = _scandir_walk(folder, extensions)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("scandir walk: %d files in %.0f ms", len(files), elapsed_ms)
    return ScanResult(files=files, method="scandir", duration_ms=elapsed_ms)


# ── scandir helpers ──────────────────────────────────────────────────────────

def _scandir_walk(folder: Path, extensions: Set[str]) -> List[Path]:
    """Recursive file enumeration using ``os.scandir`` (faster than rglob).
    
    Uses an iterative stack instead of recursion to avoid hitting
    Python's default recursion limit on deeply nested directories.
    """
    results: List[Path] = []
    stack: List[str] = [str(folder)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            ext = os.path.splitext(entry.name)[1].lower()
                            if not extensions or ext in extensions:
                                results.append(Path(entry.path))
                    except OSError:
                        continue
        except PermissionError:
            logger.debug("Permission denied: %s", current)
        except OSError as e:
            logger.debug("Skipping %s: %s", current, e)
    return results
