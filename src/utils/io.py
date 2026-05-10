"""I/O helpers: JSON, JSONL, CSV, YAML, caching."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterator

import yaml

log = logging.getLogger("pipeline.io")


# ── YAML ──────────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── JSON ──────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    log.debug(f"Wrote JSON → {path}")


# ── JSONL ─────────────────────────────────────────────────────────────────────

def append_jsonl(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    log.info(f"Wrote {len(records)} records → {path}")


# ── CSV ───────────────────────────────────────────────────────────────────────

def save_csv(records: list[dict], path: Path) -> None:
    if not records:
        log.warning(f"save_csv: no records to write to {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    log.info(f"Wrote {len(records)} rows → {path}")


# ── Disk cache ────────────────────────────────────────────────────────────────

class DiskCache:
    """
    Simple file-based cache keyed by a SHA-256 hash of the prompt.
    Prevents re-calling the LLM for identical prompts across runs.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if p.exists():
            self._hits += 1
            return load_json(p)
        self._misses += 1
        return None

    def set(self, key: str, value: dict) -> None:
        save_json(value, self._path(key))

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        rate = self._hits / total if total else 0.0
        return {"hits": self._hits, "misses": self._misses, "hit_rate": rate}
