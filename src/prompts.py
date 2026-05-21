"""
Prompt management: loads prompts from YAML and formats them with provided context.

All prompts live in configs/prompts.yaml so they can be version-tracked
independently of code and swapped without touching Python.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.io import load_yaml
from src.utils.hashing import sha256_text


_PROMPTS: dict | None = None


def _load(prompts_path: Path) -> dict:
    global _PROMPTS
    if _PROMPTS is None:
        _PROMPTS = load_yaml(prompts_path)
    return _PROMPTS


class PromptBuilder:
    """Builds (system, user) prompt pairs and tracks prompt hashes."""

    def __init__(self, prompts_path: Path) -> None:
        self._raw = _load(prompts_path)

    def _get(self, key: str) -> dict:
        if key not in self._raw:
            raise KeyError(f"Prompt key {key!r} not found in prompts.yaml")
        return self._raw[key]

    def build(self, key: str, **kwargs: object) -> tuple[str, str, str]:
        """
        Return (system_prompt, user_prompt, prompt_hash).
        kwargs are substituted into the user prompt template.
        """
        spec = self._get(key)
        system = spec["system"].strip()
        user = spec["user"].format(**kwargs).strip()
        prompt_hash = sha256_text(f"{key}|{system}|{user}")[:16]
        return system, user, prompt_hash

    # ── Convenience wrappers ──────────────────────────────────────────────────

    def profile(self, **kw: object) -> tuple[str, str, str]:
        return self.build("profile_generation", **kw)

    def profile_free(self, **kw: object) -> tuple[str, str, str]:
        """Profile generation for CSV mode: LLM chooses age_group and religion freely."""
        return self.build("profile_generation_free", **kw)

    def post(self, **kw: object) -> tuple[str, str, str]:
        return self.build("post_generation", **kw)

    def comment(self, severity: str, **kw: object) -> tuple[str, str, str]:
        key_map = {
            "opposing_opinion":  "comment_opposing_opinion",
            "dehumanising":      "comment_dehumanising",
            "inciting_violence":  "comment_inciting_violence",
        }
        if severity not in key_map:
            raise ValueError(f"Unknown severity: {severity!r}")
        return self.build(key_map[severity], **kw)

    def severity_judge(self, **kw: object) -> tuple[str, str, str]:
        return self.build("severity_judge", **kw)

    def realism_check(self, **kw: object) -> tuple[str, str, str]:
        return self.build("realism_check", **kw)
