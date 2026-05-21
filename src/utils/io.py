"""I/O helpers: JSON, JSONL, CSV, YAML, caching, topic registry."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
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


# ── Topic registry ────────────────────────────────────────────────────────────

@dataclass
class TopicMeta:
    topic_id:       str
    csv_label:      str
    prompt_label:   str          # concrete policy statement for LLM prompts
    stance_support: str          # topic-specific stance text, e.g. "support gender quotas"
    stance_oppose:  str          # topic-specific stance text, e.g. "oppose gender quotas"
    gradient:       str
    emoji:          str
    target_support: str | None = None   # optional: social group when posting in support
    target_oppose:  str | None = None   # optional: social group when posting in opposition


class TopicRegistry:
    """
    Loaded from configs/topics.yaml. Single source of truth for all topic
    metadata. Replaces the scattered TOPIC_LABELS / TARGET_GROUPS /
    TOPIC_GRADIENTS / TOPIC_EMOJIS dicts that were previously in Python code.
    """

    def __init__(self, topics_path: Path) -> None:
        raw = load_yaml(topics_path).get("topics", {})
        self._by_id:        dict[str, TopicMeta] = {}
        self._by_csv_label: dict[str, str]       = {}   # csv_label → topic_id

        for topic_id, spec in raw.items():
            tg = spec.get("target_groups") or {}
            sl = spec.get("stance_labels") or {}
            meta = TopicMeta(
                topic_id=topic_id,
                csv_label=spec["csv_label"],
                prompt_label=spec["prompt_label"],
                stance_support=sl.get("support", f"support {topic_id}"),
                stance_oppose=sl.get("oppose",   f"oppose {topic_id}"),
                gradient=spec["gradient"],
                emoji=spec["emoji"],
                target_support=tg.get("support"),
                target_oppose=tg.get("oppose"),
            )
            self._by_id[topic_id] = meta
            self._by_csv_label[spec["csv_label"]] = topic_id

    # ── Lookups ───────────────────────────────────────────────────────────────

    def topic_id_from_csv(self, csv_label: str) -> str:
        """Convert a CSV topics value to an internal topic_id. Raises KeyError on miss."""
        if csv_label not in self._by_csv_label:
            raise KeyError(
                f"Unknown CSV topic label {csv_label!r}. "
                f"Add it to configs/topics.yaml."
            )
        return self._by_csv_label[csv_label]

    def get(self, topic_id: str) -> TopicMeta:
        if topic_id not in self._by_id:
            raise KeyError(
                f"Unknown topic_id {topic_id!r}. "
                f"Add it to configs/topics.yaml."
            )
        return self._by_id[topic_id]

    def prompt_label(self, topic_id: str) -> str:
        return self.get(topic_id).prompt_label

    def stance_label(self, topic_id: str, stance: str) -> str:
        """Return the topic-specific stance text, e.g. 'support gender quotas'."""
        meta = self.get(topic_id)
        return meta.stance_support if stance == "support" else meta.stance_oppose

    def target_group(self, topic_id: str, stance: str) -> str | None:
        """Return the social group the post aligns with, or None if not configured."""
        meta = self.get(topic_id)
        return meta.target_support if stance == "support" else meta.target_oppose

    def gradient(self, topic_id: str) -> str:
        return self.get(topic_id).gradient

    def emoji(self, topic_id: str) -> str:
        return self.get(topic_id).emoji

    def all_topic_ids(self) -> list[str]:
        return list(self._by_id.keys())


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
