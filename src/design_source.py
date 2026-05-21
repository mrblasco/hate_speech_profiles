"""
Design sources: pluggable Stage 1 for the generation pipeline.

Two implementations share the same interface:
  CsvDesignSource  — reads an R-generated CSV (one file per country)
  InternalDesignSource — wraps the existing balanced factorial sampler

Both produce DesignRow objects that the pipeline consumes.  The CSV source
additionally carries per-respondent metadata (respondent_id, anonymity,
popularity) used in the output CSV for R join-back.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from src.models import CommentSeverity, Gender
from src.sampling import Condition, build_design_matrix
from src.utils.io import TopicRegistry

log = logging.getLogger("pipeline.design_source")

_POPULARITY_MAP = {
    "low":  "ordinary user",
    "mid":  "active user",
    "high": "micro-influencer",
}

_SEVERITY_MAP = {
    "low":  CommentSeverity.opposing_opinion,
    "mid":  CommentSeverity.dehumanising,
    "high": CommentSeverity.inciting_violence,
}

_COUNTRY_KEYWORDS = {
    "belgium": "Belgium",
    "italy":   "Italy",
    "spain":   "Spain",
    "france":  "France",
    "germany": "Germany",
}


# ── Core data structure ───────────────────────────────────────────────────────

@dataclass
class DesignRow:
    """
    One respondent task from the design matrix.

    For internal-sampler mode, respondent_id / anonymity / popularity are
    empty and severity comes from generating all three levels per post.
    For CSV mode, all fields are populated from the input file.
    """
    profile_id:    str
    topic:         str              # internal topic_id (e.g. "feminism")
    stance:        str              # "support" | "oppose"
    gender:        str              # "male" | "female"
    popularity:    str              # "ordinary user" | "active user" | "micro-influencer"
    severity:      CommentSeverity
    anonymity:     str              # "named" | "anonymous"
    respondent_id: str = ""
    country:       str = ""
    # original integer keys from R design (for join-back)
    topic_id:      int = 0
    stance_id:     int = 0
    severity_id:   int = 0


# ── Protocol ──────────────────────────────────────────────────────────────────

class DesignSource(Protocol):
    """
    Common interface for design sources.  Stage 1 of the pipeline calls
    `conditions()` to obtain profile-generation inputs, and `rows()` to
    access the full per-respondent task list for assembly and output.
    """

    def conditions(self) -> list[Condition]:
        """Unique profile conditions (one per profile to generate)."""
        ...

    def rows(self) -> list[DesignRow]:
        """Full list of respondent tasks (may be larger than conditions)."""
        ...


# ── CSV implementation ────────────────────────────────────────────────────────

def _detect_country(path: Path) -> str:
    stem = path.stem.lower()
    for keyword, country in _COUNTRY_KEYWORDS.items():
        if keyword in stem:
            return country
    return ""


class CsvDesignSource:
    """
    Loads an R-generated stimulus design CSV.

    One CSV file covers one country.  country_of_origin is either passed
    explicitly or auto-detected from the filename stem.

    After loading, `conditions()` returns the 6 unique profile conditions
    (deduped by profile_id).  `rows()` returns all 2000 respondent tasks.
    """

    def __init__(
        self,
        path: Path,
        topic_registry: TopicRegistry,
        country: str = "",
    ) -> None:
        self._path     = path
        self._registry = topic_registry
        self._country  = country or _detect_country(path)

        if not self._country:
            raise ValueError(
                f"Cannot determine country for {path.name}. "
                "Either name the file with the country (e.g. stim_df_italy.csv) "
                "or pass --country explicitly."
            )

        self._rows: list[DesignRow] = []
        self._conditions: list[Condition] = []
        self._load()

    # ── DesignSource interface ────────────────────────────────────────────────

    def conditions(self) -> list[Condition]:
        return self._conditions

    def rows(self) -> list[DesignRow]:
        return self._rows

    @property
    def country(self) -> str:
        return self._country

    # ── Internal ─────────────────────────────────────────────────────────────

    def _load(self) -> None:
        raw_rows = self._read_csv()
        self._rows = [self._parse_row(r) for r in raw_rows]
        self._conditions = self._build_conditions()
        log.info(
            "CSV design: %d respondent tasks, %d unique profiles, country=%s",
            len(self._rows), len(self._conditions), self._country,
        )

    def _read_csv(self) -> list[dict]:
        with open(self._path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _parse_row(self, r: dict) -> DesignRow:
        csv_label = r["topics"].strip()
        try:
            topic_id = self._registry.topic_id_from_csv(csv_label)
        except KeyError as exc:
            raise ValueError(
                f"Row respondent_id={r.get('respondent_id')}: {exc}"
            ) from exc

        stance = r["stance"].strip().lower()
        if stance not in ("support", "oppose"):
            raise ValueError(f"Unknown stance {stance!r} — expected support or oppose.")

        likes = r["likes"].strip().lower()
        severity_raw = r["severity"].strip().lower()

        if severity_raw not in _SEVERITY_MAP:
            raise ValueError(
                f"Unknown severity {severity_raw!r} — expected low, mid, or high."
            )

        return DesignRow(
            respondent_id=r["respondent_id"].strip(),
            profile_id=r["profile_id"].strip(),
            topic=topic_id,
            stance=stance,
            gender=r["gender"].strip().lower(),
            popularity=_POPULARITY_MAP.get(likes, "active user"),
            severity=_SEVERITY_MAP[severity_raw],
            anonymity=r["anonymity"].strip().lower(),
            country=self._country,
            topic_id=int(r.get("topic_id", 0)),
            stance_id=int(r.get("stance_id", 0)),
            severity_id=int(r.get("severity_id", 0)),
        )

    def _build_conditions(self) -> list[Condition]:
        """
        Build one Condition per unique profile_id.

        The profile persona depends on gender and popularity but is
        independent of topic/stance (which vary across the same profile's
        tasks).  Topic, stance, age_group, and religion are left blank;
        generate_profile() uses profile_generation_free when
        popularity_level is set.
        """
        seen: dict[str, Condition] = {}
        for row in self._rows:
            if row.profile_id not in seen:
                seen[row.profile_id] = Condition(
                    profile_id=f"CSV_{row.profile_id:0>4}",
                    topic="",
                    age_group="",
                    gender=row.gender,
                    stance="",
                    religion="",
                    country_of_origin=self._country,
                    popularity_level=row.popularity,
                )
        return list(seen.values())

    # ── Helpers for the pipeline ──────────────────────────────────────────────

    def unique_post_conditions(
        self, profile_map: dict[str, object]
    ) -> list[tuple[object, Condition]]:
        """
        Unique (profile, condition) pairs for post generation,
        deduped by (profile_id, topic, stance).
        """
        seen: set[tuple[str, str, str]] = set()
        result = []
        for row in self._rows:
            key = (row.profile_id, row.topic, row.stance)
            if key not in seen:
                seen.add(key)
                internal_pid = f"CSV_{row.profile_id:0>4}"
                profile = profile_map.get(internal_pid)
                if profile is None:
                    continue
                cond = Condition(
                    profile_id=internal_pid,
                    topic=row.topic,
                    age_group=getattr(profile, "age_group", "young_adult").value
                              if hasattr(getattr(profile, "age_group", ""), "value")
                              else str(getattr(profile, "age_group", "young_adult")),
                    gender=row.gender,
                    stance=row.stance,
                    religion="",
                    country_of_origin=self._country,
                    popularity_level=row.popularity,
                )
                result.append((profile, cond))
        return result

    def comment_severities_for_post(self, post_id: str) -> list[CommentSeverity]:
        """Which severity levels to generate for a given post_id."""
        # post_id format: CSV_{profile_id}_{topic}_{stance}_POST00
        # Map back via rows
        seen: set[CommentSeverity] = set()
        for row in self._rows:
            if self._post_id_for_row(row) == post_id:
                seen.add(row.severity)
        return list(seen)

    @staticmethod
    def _post_id_for_row(row: DesignRow) -> str:
        return f"CSV_{row.profile_id:0>4}_{row.topic}_{row.stance}_POST00"


# ── Internal sampler implementation ───────────────────────────────────────────

class InternalDesignSource:
    """
    Wraps the existing balanced factorial sampler.
    Preserves all current pipeline behaviour for non-CSV runs.
    """

    def __init__(
        self,
        study_cfg: dict,
        n_profiles: int,
        seed: int,
    ) -> None:
        design = study_cfg.get("design", {})
        # Map legacy "progressive"/"conservative" values to support/oppose
        raw_stances = design.get("values", ["support", "oppose"])
        stances = [
            "support" if s == "progressive" else
            "oppose"  if s == "conservative" else s
            for s in raw_stances
        ]
        self._conds = build_design_matrix(
            n_profiles=n_profiles,
            topics=design.get("topics", []),
            age_groups=design.get("age_groups", []),
            genders=design.get("genders", []),
            stances=stances,
            religions=design.get("religions", []),
            countries_of_origin=design.get("countries_of_origin", []),
            seed=seed,
        )

    def conditions(self) -> list[Condition]:
        return self._conds

    def rows(self) -> list[DesignRow]:
        # Internal mode: rows mirror conditions one-to-one
        return [
            DesignRow(
                profile_id=c.profile_id,
                topic=c.topic,
                stance=c.stance,
                gender=c.gender,
                popularity=c.popularity_level or "active user",
                severity=CommentSeverity.opposing_opinion,  # placeholder; pipeline generates all 3
                anonymity="named",
                respondent_id="",
                country=c.country_of_origin,
            )
            for c in self._conds
        ]
