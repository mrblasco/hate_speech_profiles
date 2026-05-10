"""
Schema validation and content rule enforcement.

Validates generated items against:
  A. Pydantic schema correctness (already enforced in generators, rechecked here)
  B. Domain rules from generation_rules.yaml (word counts, forbidden patterns)
  C. Deduplication (hash-based exact + fuzzy similarity)
"""

from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path

from src.models import Comment, CommentSeverity, OriginalPost, Profile
from src.utils.hashing import sha256_text
from src.utils.io import load_yaml

log = logging.getLogger("pipeline.validators.schema")


class SchemaValidator:
    """Validates generated objects against domain rules."""

    def __init__(self, rules_path: Path) -> None:
        rules = load_yaml(rules_path)
        self._post_rules    = rules.get("post_rules", {})
        self._comment_rules = rules.get("comment_rules", {})
        self._dedup_threshold = 0.85

        self._forbidden: list[re.Pattern] = [
            re.compile(rf'\b{re.escape(p)}\b', re.IGNORECASE)
            for p in self._post_rules.get("forbidden_patterns", [])
        ]

        # Deduplication state
        self._seen_post_hashes:    set[str] = set()
        self._seen_comment_hashes: set[str] = set()
        self._seen_post_texts:     list[str] = []
        self._seen_comment_texts:  list[str] = []

    # ── Public validators ─────────────────────────────────────────────────────

    def validate_post(self, post: OriginalPost) -> list[str]:
        """Return list of violation strings (empty = pass)."""
        issues: list[str] = []

        min_w = self._post_rules.get("min_words", 15)
        max_w = self._post_rules.get("max_words", 40)
        if not (min_w <= post.word_count <= max_w):
            issues.append(
                f"word_count={post.word_count} outside [{min_w}, {max_w}]"
            )

        for pat in self._forbidden:
            if pat.search(post.caption):
                issues.append(f"Forbidden pattern found: {pat.pattern!r}")

        # Deduplication
        h = sha256_text(post.caption.lower())
        if h in self._seen_post_hashes:
            issues.append("Exact duplicate caption detected")
        else:
            if self._is_fuzzy_duplicate(post.caption, self._seen_post_texts):
                issues.append("Near-duplicate caption detected")
            else:
                self._seen_post_hashes.add(h)
                self._seen_post_texts.append(post.caption)

        return issues

    def validate_comment(self, comment: Comment) -> list[str]:
        """Return list of violation strings (empty = pass)."""
        issues: list[str] = []

        rules = self._comment_rules.get(comment.severity.value, {})

        # Toxicity range checks
        if "max_toxicity_estimate" in rules:
            if comment.toxicity_estimate > rules["max_toxicity_estimate"]:
                issues.append(
                    f"toxicity {comment.toxicity_estimate:.2f} exceeds max "
                    f"{rules['max_toxicity_estimate']:.2f} for {comment.severity.value}"
                )
        if "min_toxicity_estimate" in rules:
            if comment.toxicity_estimate < rules["min_toxicity_estimate"]:
                issues.append(
                    f"toxicity {comment.toxicity_estimate:.2f} below min "
                    f"{rules['min_toxicity_estimate']:.2f} for {comment.severity.value}"
                )
        if "toxicity_estimate_range" in rules:
            lo, hi = rules["toxicity_estimate_range"]
            if not (lo <= comment.toxicity_estimate <= hi):
                issues.append(
                    f"toxicity {comment.toxicity_estimate:.2f} outside "
                    f"[{lo:.2f}, {hi:.2f}] for {comment.severity.value}"
                )

        # Violence flag check
        if "contains_explicit_violence" in rules:
            expected = rules["contains_explicit_violence"]
            if comment.contains_explicit_violence != expected:
                issues.append(
                    f"contains_explicit_violence should be {expected} "
                    f"for {comment.severity.value}"
                )

        # Deduplication
        h = sha256_text(comment.text.lower())
        if h in self._seen_comment_hashes:
            issues.append("Exact duplicate comment text detected")
        else:
            if self._is_fuzzy_duplicate(comment.text, self._seen_comment_texts):
                issues.append("Near-duplicate comment detected")
            else:
                self._seen_comment_hashes.add(h)
                self._seen_comment_texts.append(comment.text)

        return issues

    def validate_profile(self, profile: Profile) -> list[str]:
        """Lightweight profile checks beyond Pydantic constraints."""
        issues: list[str] = []
        if len(profile.bio) > 150:
            issues.append(f"bio too long: {len(profile.bio)} chars")
        if len(profile.interests) < 1:
            issues.append("interests list is empty")
        return issues

    # ── Internal ──────────────────────────────────────────────────────────────

    def _is_fuzzy_duplicate(self, text: str, corpus: list[str]) -> bool:
        if not corpus:
            return False
        text_lower = text.lower()
        for existing in corpus[-200:]:   # check against last 200 to bound cost
            ratio = difflib.SequenceMatcher(
                None, text_lower, existing.lower(), autojunk=False
            ).ratio()
            if ratio >= self._dedup_threshold:
                return True
        return False
