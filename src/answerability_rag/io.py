"""Checksum-verified downloads, safe extraction, and canonical artifact writes."""

from __future__ import annotations

import csv
import json
import os
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .hashing import canonical_json, sha256_file
from .hashing import canonical_json_sha256


@dataclass(frozen=True)
class FileRecord:
    path: str
    url: str
    resolved_url: str
    bytes: int
    sha256: str
    retrieved_at: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def download_verified(url: str, destination: Path, expected_sha256: str | None) -> FileRecord:
    """Download without exposing partial files; accept an identical cached file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        observed = sha256_file(destination)
        if expected_sha256 is not None and observed != expected_sha256:
            raise ValueError(f"refusing mismatched existing file {destination}: {observed}")
        return FileRecord(str(destination), url, url, destination.stat().st_size, observed, "cached")

    handle, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".part", dir=destination.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    resolved_url = url
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "answerability-rag-phase01/1"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            resolved_url = response.geturl()
            shutil.copyfileobj(response, output, length=1024 * 1024)
        observed = sha256_file(temporary)
        if expected_sha256 is not None and observed != expected_sha256:
            raise ValueError(f"checksum mismatch for {url}: expected {expected_sha256}, got {observed}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return FileRecord(
        str(destination), url, resolved_url, destination.stat().st_size,
        sha256_file(destination), utc_now(),
    )


def safe_extract_zip(archive: Path, destination: Path) -> list[Path]:
    """Extract regular files only, rejecting traversal, links, and changed reruns."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    extracted: list[Path] = []
    marker = destination.parent / f".{destination.name}_extraction.json"
    with zipfile.ZipFile(archive) as bundle:
        members = [member for member in bundle.infolist() if not member.is_dir()]
        member_signature = canonical_json_sha256([
            {"path": member.filename, "bytes": member.file_size, "crc32": member.CRC}
            for member in members
        ])
        expected_marker = {
            "archive_sha256": sha256_file(archive), "member_count": len(members),
            "member_signature_sha256": member_signature,
        }
        if marker.exists():
            observed_marker = json.loads(marker.read_text(encoding="utf-8"))
            targets = [destination.joinpath(*PurePosixPath(member.filename).parts) for member in members]
            if observed_marker == expected_marker and all(
                target.is_file() and target.stat().st_size == member.file_size
                for target, member in zip(targets, members)
            ):
                return sorted(targets, key=lambda path: path.relative_to(destination).as_posix())
            raise ValueError(f"extraction marker/cache mismatch under {destination}")
        for member in members:
            pure = PurePosixPath(member.filename)
            if member.is_dir():
                continue
            mode = member.external_attr >> 16
            file_type = stat.S_IFMT(mode)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                raise ValueError(f"unsafe ZIP path: {member.filename!r}")
            if stat.S_ISLNK(mode) or (file_type and file_type != stat.S_IFREG):
                raise ValueError(f"ZIP member is not a regular file: {member.filename!r}")
            target = destination.joinpath(*pure.parts)
            resolved = target.resolve()
            if root != resolved and root not in resolved.parents:
                raise ValueError(f"ZIP member escapes destination: {member.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source:
                incoming = source.read()
            if target.exists():
                if target.read_bytes() != incoming:
                    raise ValueError(f"refusing to overwrite changed extracted file: {target}")
            else:
                target.write_bytes(incoming)
            extracted.append(target)
    marker.write_text(json.dumps(expected_marker, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return sorted(extracted, key=lambda path: path.relative_to(destination).as_posix())


def csv_bytes(rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: "" if row.get(name) is None else row.get(name) for name in fieldnames})
    return output.getvalue().encode("utf-8")


def write_bytes_atomic(path: Path, content: bytes, *, immutable: bool = False) -> str:
    """Atomically write; immutable artifacts may only be recreated byte-identically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return sha256_file(path)
        if immutable:
            raise ValueError(f"frozen artifact differs from regenerated content: {path}")
    handle, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        Path(temporary_name).replace(path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def write_csv_atomic(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str],
                     *, immutable: bool = False) -> str:
    return write_bytes_atomic(path, csv_bytes(rows, fieldnames), immutable=immutable)


def write_json_atomic(path: Path, value: Any, *, immutable: bool = False) -> str:
    content = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    return write_bytes_atomic(path, content, immutable=immutable)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_json_field(value: Any) -> str:
    return canonical_json(value)
