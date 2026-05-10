"""Deterministic hashing utilities for deduplication and provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dict(d: dict) -> str:
    serialised = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return sha256_text(serialised)


def short_hash(text: str, length: int = 8) -> str:
    return sha256_text(text)[:length]
