"""
Pydantic schemas for all generated and validated objects.

Every model carries enough metadata to fully reconstruct how it was produced,
satisfying the reproducibility requirements of the study.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class Topic(str, Enum):
    immigration      = "immigration"
    feminism         = "feminism"
    religion         = "religion"
    gender          = "gender"
    racism          = "racism"
    climate          = "climate"
    public_health    = "public_health"
    national_identity = "national_identity"


class AgeGroup(str, Enum):
    adolescent  = "adolescent"
    young_adult = "young_adult"


class Gender(str, Enum):
    male      = "male"
    female    = "female"
    nonbinary = "nonbinary"


class Values(str, Enum):
    progressive  = "progressive"
    conservative = "conservative"


class CommentSeverity(str, Enum):
    opposing_opinion  = "opposing_opinion"
    dehumanising      = "dehumanising"
    inciting_violence = "inciting_violence"


class Religion(str, Enum):
    muslim    = "Muslim"
    christian = "Christian"
    jewish    = "Jewish"


class CountryOfOrigin(str, Enum):
    belgium = "Belgium"
    italy  = "Italy"
    spain  = "Spain"
    france = "France"
    germany = "Germany"


# ── Core domain models ────────────────────────────────────────────────────────

class Profile(BaseModel):
    profile_id:        str
    username:          str
    display_name:      str
    age:               int = Field(ge=13, le=25)
    age_group:         AgeGroup
    gender:            Gender
    values:            Values
    religion:          Religion
    country_of_origin: CountryOfOrigin
    interests:         list[str] = Field(min_length=1, max_length=5)
    writing_style:     str
    bio:               str = Field(max_length=150)

    @field_validator("username")
    @classmethod
    def username_valid(cls, v: str) -> str:
        import re
        if not re.match(r'^[a-z0-9._]{1,30}$', v):
            raise ValueError(f"Invalid Instagram username: {v!r}")
        return v

    @field_validator("age_group", mode="before")
    @classmethod
    def coerce_age_group(cls, v: str) -> str:
        return v.lower().replace(" ", "_").replace("-", "_")


class OriginalPost(BaseModel):
    post_id:    str
    profile_id: str
    topic:      Topic
    caption:    str
    hashtags:   list[str] = Field(default_factory=list)
    word_count: int = Field(ge=1)

    @field_validator("word_count", mode="before")
    @classmethod
    def compute_word_count(cls, v: object, info: object) -> int:
        # Accept provided value but allow recomputation during validation
        return int(v) if v is not None else 0

    @model_validator(mode="after")
    def validate_word_count(self) -> "OriginalPost":
        actual = len(self.caption.split())
        if self.word_count != actual:
            self.word_count = actual
        return self


class Comment(BaseModel):
    comment_id:               str
    post_id:                  str
    severity:                 CommentSeverity
    text:                     str
    toxicity_estimate:        float = Field(ge=0.0, le=1.0)
    target_group:             str
    contains_explicit_violence: bool


# ── Judge / validation models ─────────────────────────────────────────────────

class SeverityJudgement(BaseModel):
    severity_score: int = Field(ge=1, le=3)
    severity_label: CommentSeverity
    confidence:     float = Field(ge=0.0, le=1.0)
    reasoning:      str


class RealismCheck(BaseModel):
    is_realistic:   bool
    realism_score:  float = Field(ge=0.0, le=1.0)
    issues:         list[str] = Field(default_factory=list)


# ── Generation metadata ───────────────────────────────────────────────────────

class GenerationMeta(BaseModel):
    """Provenance record attached to every generated item."""
    model_name:   str
    prompt_text:  str
    prompt_hash:  str = ""          # populated automatically
    temperature:  float
    seed:         int
    timestamp:    datetime = Field(default_factory=datetime.utcnow)
    run_id:       str = ""
    experiment_id: str = ""

    def model_post_init(self, __context: object) -> None:
        if not self.prompt_hash and self.prompt_text:
            self.prompt_hash = hashlib.sha256(
                self.prompt_text.encode()
            ).hexdigest()[:16]


# ── Flat stimulus row (one row per comment × post combination) ────────────────

class StimulusRow(BaseModel):
    """
    Fully denormalized row written to the final CSV / JSONL output.
    One row = one experimental stimulus cell.
    """
    # Condition identifiers
    stimulus_id:   str
    experiment_id: str
    run_id:        str

    # Profile fields
    profile_id:        str
    username:          str
    display_name:      str
    age:               int
    age_group:         AgeGroup
    gender:            Gender
    values:            Values
    religion:          Optional[Religion]          = None
    country_of_origin: Optional[CountryOfOrigin]  = None
    writing_style:     str
    bio:               str

    # Post fields
    post_id:       str
    topic:         Topic
    caption:       str
    hashtags:      str          # pipe-joined for CSV compatibility
    post_word_count: int

    # Comment fields
    comment_id:              str
    severity:                CommentSeverity
    comment_text:            str
    toxicity_estimate:       float
    target_group:            str
    contains_explicit_violence: bool

    # Policy fields (None in standard topic-based runs)
    policy_id:       Optional[str] = None
    post_stance:     Optional[str] = None
    opposing_stance: Optional[str] = None

    # Validation fields
    judge_severity_score:    Optional[int]   = None
    judge_severity_label:    Optional[str]   = None
    judge_confidence:        Optional[float] = None
    judge_agrees:            Optional[bool]  = None
    realism_score:           Optional[float] = None
    passed_validation:       bool = True

    # Provenance
    model_name:   str
    prompt_hash:  str
    temperature:  float
    seed:         int
    timestamp:    datetime

    @classmethod
    def from_parts(
        cls,
        profile: Profile,
        post: OriginalPost,
        comment: Comment,
        meta: GenerationMeta,
        judge: Optional[SeverityJudgement] = None,
        realism: Optional[RealismCheck] = None,
        passed: bool = True,
        policy_id: Optional[str] = None,
        post_stance: Optional[str] = None,
        opposing_stance: Optional[str] = None,
    ) -> "StimulusRow":
        stimulus_id = f"{post.post_id}_{comment.severity.value}"
        return cls(
            stimulus_id=stimulus_id,
            experiment_id=meta.experiment_id,
            run_id=meta.run_id,
            profile_id=profile.profile_id,
            username=profile.username,
            display_name=profile.display_name,
            age=profile.age,
            age_group=profile.age_group,
            gender=profile.gender,
            values=profile.values,
            religion=profile.religion,
            country_of_origin=profile.country_of_origin,
            writing_style=profile.writing_style,
            bio=profile.bio,
            post_id=post.post_id,
            topic=post.topic,
            caption=post.caption,
            hashtags="|".join(post.hashtags),
            post_word_count=post.word_count,
            comment_id=comment.comment_id,
            severity=comment.severity,
            comment_text=comment.text,
            toxicity_estimate=comment.toxicity_estimate,
            target_group=comment.target_group,
            contains_explicit_violence=comment.contains_explicit_violence,
            judge_severity_score=judge.severity_score if judge else None,
            judge_severity_label=judge.severity_label.value if judge else None,
            judge_confidence=judge.confidence if judge else None,
            judge_agrees=(
                judge.severity_label == comment.severity if judge else None
            ),
            realism_score=realism.realism_score if realism else None,
            passed_validation=passed,
            policy_id=policy_id,
            post_stance=post_stance,
            opposing_stance=opposing_stance,
            model_name=meta.model_name,
            prompt_hash=meta.prompt_hash,
            temperature=meta.temperature,
            seed=meta.seed,
            timestamp=meta.timestamp,
        )


# ── Run manifest ──────────────────────────────────────────────────────────────

class RunManifest(BaseModel):
    """Top-level metadata written to generation_metadata.json."""
    experiment_id:     str
    run_id:            str
    timestamp_start:   datetime
    timestamp_end:     Optional[datetime] = None
    seed:              int
    n_profiles:        int
    n_posts:           int
    n_comments:        int
    n_stimuli:         int
    n_passed:          int
    n_failed:          int
    model_name:        str
    temperature:       float
    config_hash:       str       # SHA-256 of study_config.yaml at run time
    topics:            list[str]
    age_groups:        list[str]
    genders:           list[str]
    values:            list[str]
    severities:        list[str]
    output_files:      dict[str, str] = Field(default_factory=dict)
