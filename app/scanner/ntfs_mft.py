"""
NTFS Master File Table (MFT) scanner for ultra-fast file enumeration.

Reads the MFT directly via Windows DeviceIoControl (FSCTL_ENUM_USN_DATA),
bypassing recursive directory traversal.  Can enumerate millions of files
in seconds — the same technique used by TreeSize and Everything.

Requirements
------------
- Windows OS
- NTFS file system on the target volume
- Administrator privileges (needed to open the volume handle)
"""

import ctypes
import ctypes.wintypes as wintypes
import struct
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ── Win32 constants ──────────────────────────────────────────────────────────
GENERIC_READ            = 0x80000000
FILE_SHARE_READ         = 0x00000001
FILE_SHARE_WRITE        = 0x00000002
OPEN_EXISTING           = 3
INVALID_HANDLE_VALUE    = ctypes.c_void_p(-1).value
FSCTL_ENUM_USN_DATA     = 0x000900B3
FILE_ATTRIBUTE_DIRECTORY = 0x00000010

# NTFS root directory always has MFT reference number 5
NTFS_ROOT_REF = 5

# Buffer size for MFT enumeration reads
_MFT_BUF_SIZE = 128 * 1024  # 128 KB

import platform as _platform
if _platform.system() == "Windows":
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
else:
    kernel32 = None  # type: ignore[assignment]


@dataclass
class MFTEntry:
    """A single Master File Table entry (file or directory)."""
    file_ref: int
    parent_ref: int
    name: str
    is_dir: bool


class NTFSScanner:
    """
    Reads the NTFS MFT to enumerate files without directory traversal.

    Usage::

        scanner = NTFSScanner()
        paths = scanner.scan_folder(Path("D:/Projects"), {".txt", ".md"})
        if paths is None:
            # Fall back to os.scandir
    """

    def __init__(self) -> None:
        if kernel32 is None:
            raise RuntimeError("NTFSScanner requires Windows (kernel32 not available)")
        self._k32 = kernel32
        self.entries: Dict[int, MFTEntry] = {}
        self.children_map: Dict[int, List[MFTEntry]] = defaultdict(list)
        self.entry_count: int = 0

    # ── public API ───────────────────────────────────────────────────────

    def scan_folder(
        self,
        folder: Path,
        extensions: Set[str],
    ) -> Optional[List[Path]]:
        """
        Enumerate files under *folder* whose suffix is in *extensions*.

        Returns ``None`` when MFT access is unavailable (e.g. not admin,
        not NTFS).  The caller should fall back to ``os.scandir``.
        """
        volume = folder.drive  # e.g. "C:"
        if not volume:
            logger.warning("Cannot determine volume for path: %s", folder)
            return None

        handle = self._open_volume(volume)
        if handle == INVALID_HANDLE_VALUE:
            err = ctypes.get_last_error()
            logger.info(
                "Cannot open volume %s (Win32 error %d). "
                "Run as Administrator for NTFS-accelerated scanning.",
                volume,
                err,
            )
            return None

        try:
            t0 = time.perf_counter()
            self._enumerate_mft(handle)
            elapsed = time.perf_counter() - t0
            self.entry_count = len(self.entries)
            logger.info(
                "MFT enumeration: %s entries in %.2fs",
                f"{self.entry_count:,}",
                elapsed,
            )
        finally:
            self._k32.CloseHandle(handle)

        # Build parent → children index
        self._build_children_map()

        # Walk the tree to the target folder
        target_ref = self._find_folder_ref(folder)
        if target_ref is None:
            logger.warning("Folder not found in MFT: %s", folder)
            return None

        # Collect matching files under the target folder (BFS)
        file_paths = self._collect_files(target_ref, extensions, folder)
        logger.info(
            "MFT scan: %d matching files in %s",
            len(file_paths),
            folder,
        )
        return file_paths

    # ── internals ────────────────────────────────────────────────────────

    def _open_volume(self, volume: str) -> int:
        """Open a raw volume handle (requires admin)."""
        return self._k32.CreateFileW(
            f"\\\\.\\{volume}",
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            None,
            OPEN_EXISTING,
            0,
            None,
        )

    def _enumerate_mft(self, handle: int) -> None:
        """
        Read every MFT entry via ``FSCTL_ENUM_USN_DATA``.

        The input buffer is an ``MFT_ENUM_DATA_V0`` struct::

            StartFileReferenceNumber  DWORDLONG   (8 bytes)
            LowUsn                    LONGLONG    (8 bytes)
            HighUsn                   LONGLONG    (8 bytes)

        The output buffer starts with the *next* file-reference number
        (8 bytes), followed by a stream of ``USN_RECORD_V2`` entries.
        """
        # MFT_ENUM_DATA_V0: start=0, low=0, high=MAX
        med = struct.pack("<Qqq", 0, 0, 0x7FFFFFFFFFFFFFFF)

        buf_size = _MFT_BUF_SIZE
        buf = ctypes.create_string_buffer(buf_size)
        bytes_returned = wintypes.DWORD()

        self.entries.clear()

        while True:
            ok = self._k32.DeviceIoControl(
                handle,
                FSCTL_ENUM_USN_DATA,
                med,
                len(med),
                buf,
                buf_size,
                ctypes.byref(bytes_returned),
                None,
            )
            if not ok:
                err = ctypes.get_last_error()
                if err:
                    logger.debug("DeviceIoControl ended (Win32 error %d)", err)
                break

            returned = bytes_returned.value
            if returned <= 8:
                break

            # First 8 bytes → next StartFileReferenceNumber
            next_ref = struct.unpack_from("<Q", buf.raw, 0)[0]

            # Parse USN_RECORD_V2 entries starting at offset 8
            # ── USN_RECORD_V2 layout ──
            #  0  RecordLength          DWORD   4
            #  4  MajorVersion          WORD    2
            #  6  MinorVersion          WORD    2
            #  8  FileReferenceNumber   QWORD   8
            # 16  ParentFileRefNumber   QWORD   8
            # 24  Usn                   QWORD   8
            # 32  TimeStamp             QWORD   8
            # 40  Reason                DWORD   4
            # 44  SourceInfo            DWORD   4
            # 48  SecurityId            DWORD   4
            # 52  FileAttributes        DWORD   4
            # 56  FileNameLength        WORD    2
            # 58  FileNameOffset        WORD    2
            # 60+ FileName              WCHAR[]
            offset = 8
            while offset + 60 <= returned:
                rec_len = struct.unpack_from("<I", buf.raw, offset)[0]
                if rec_len == 0 or offset + rec_len > returned:
                    break

                file_ref = (
                    struct.unpack_from("<Q", buf.raw, offset + 8)[0]
                    & 0x0000FFFFFFFFFFFF
                )
                parent_ref = (
                    struct.unpack_from("<Q", buf.raw, offset + 16)[0]
                    & 0x0000FFFFFFFFFFFF
                )
                attrs = struct.unpack_from("<I", buf.raw, offset + 52)[0]
                name_len = struct.unpack_from("<H", buf.raw, offset + 56)[0]
                name_off = struct.unpack_from("<H", buf.raw, offset + 58)[0]

                name_start = offset + name_off
                name_end = name_start + name_len
                if name_end <= returned:
                    name = buf.raw[name_start:name_end].decode(
                        "utf-16-le", errors="replace"
                    )
                    is_dir = bool(attrs & FILE_ATTRIBUTE_DIRECTORY)
                    self.entries[file_ref] = MFTEntry(
                        file_ref=file_ref,
                        parent_ref=parent_ref,
                        name=name,
                        is_dir=is_dir,
                    )

                offset += rec_len

            # Advance to next batch
            med = struct.pack("<Qqq", next_ref, 0, 0x7FFFFFFFFFFFFFFF)

    def _build_children_map(self) -> None:
        """Index every entry by its parent reference."""
        self.children_map.clear()
        for entry in self.entries.values():
            self.children_map[entry.parent_ref].append(entry)

    def _find_folder_ref(self, folder: Path) -> Optional[int]:
        """
        Walk from the NTFS root (ref 5) down the path components
        to find the target folder's MFT reference number.
        """
        # folder.parts → ('D:\\', 'Projects', 'Work')  →  skip drive root
        parts = folder.parts[1:]
        current_ref = NTFS_ROOT_REF

        for part in parts:
            part_lower = part.lower()
            found = False
            for child in self.children_map.get(current_ref, []):
                if child.is_dir and child.name.lower() == part_lower:
                    current_ref = child.file_ref
                    found = True
                    break
            if not found:
                return None

        return current_ref

    def _collect_files(
        self,
        folder_ref: int,
        extensions: Set[str],
        base_path: Path,
    ) -> List[Path]:
        """
        BFS from *folder_ref*, building full paths top-down.

        Only files whose extension is in *extensions* are returned.
        """
        results: List[Path] = []
        stack = [(folder_ref, base_path)]

        while stack:
            ref, current_path = stack.pop()
            for child in self.children_map.get(ref, []):
                child_path = current_path / child.name
                if child.is_dir:
                    stack.append((child.file_ref, child_path))
                else:
                    if not extensions or child_path.suffix.lower() in extensions:
                        results.append(child_path)

        return results
