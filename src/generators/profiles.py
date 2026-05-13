"""
Profile generator: creates fictional Instagram profiles via LLM.

Each profile is generated independently with a derived seed so results are
reproducible and auditable. Profiles are validated against the Profile schema
before being returned.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from src.llm_client import LLMClient
from src.models import Profile
from src.prompts import PromptBuilder
from src.sampling import Condition
from src.utils.seeds import derive_seed

log = logging.getLogger("pipeline.generators.profiles")

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "study_config.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as _f:
    _age_ranges_cfg = yaml.safe_load(_f)["design"].get("age_ranges", {})

AGE_RANGES: dict[str, tuple[int, int]] = {
    k: (v["min"], v["max"]) for k, v in _age_ranges_cfg.items()
}


async def generate_profile(
    condition: Condition,
    client: LLMClient,
    prompt_builder: PromptBuilder,
    base_seed: int,
) -> tuple[Profile, str]:
    """
    Generate and validate one Profile.

    Returns (profile, prompt_hash).
    Raises ValueError if schema validation fails after all retries.
    """
    f = condition.factors
    age_group       = f.get("target_age_group", "young_adult")
    gender          = f.get("target_gender", "")
    religion        = f.get("target_religion", "")
    country_of_origin = f.get("target_origin", "")
    topic           = f.get("post_topic", "")

    age_min, age_max = AGE_RANGES.get(age_group, (18, 25))
    seed = derive_seed(base_seed, "profile", condition.profile_id)

    system, user, prompt_hash = prompt_builder.profile(
        profile_id=condition.profile_id,
        topic=topic,
        age_group=age_group,
        age_min=age_min,
        age_max=age_max,
        gender=gender,
        religion=religion,
        country_of_origin=country_of_origin,
    )

    log.debug("Generating profile %s (topic=%s, %s/%s)",
              condition.profile_id, topic, age_group, gender)

    raw = await client.complete_json(system, user, seed=seed)

    raw.setdefault("profile_id",        condition.profile_id)
    raw.setdefault("age_group",         age_group)
    raw.setdefault("gender",            gender)
    raw.setdefault("religion",          religion)
    raw.setdefault("country_of_origin", country_of_origin)

    try:
        profile = Profile.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(
            f"Profile schema validation failed for {condition.profile_id}: {exc}"
        ) from exc

    log.info("  ✓ Profile %s  @%s  (%s, %s)",
             profile.profile_id, profile.username,
             profile.gender.value, profile.age_group.value)

    return profile, prompt_hash


async def generate_profiles_batch(
    conditions: list[Condition],
    client: LLMClient,
    prompt_builder: PromptBuilder,
    base_seed: int,
) -> list[tuple[Profile, Condition, str]]:
    """
    Generate all profiles concurrently (bounded by client semaphore).
    Returns list of (profile, condition, prompt_hash).
    """
    import asyncio

    async def _one(cond: Condition) -> tuple[Profile, Condition, str]:
        profile, ph = await generate_profile(cond, client, prompt_builder, base_seed)
        return profile, cond, ph

    tasks = [asyncio.create_task(_one(c)) for c in conditions]
    results: list[tuple[Profile, Condition, str]] = []
    errors = 0

    for coro in asyncio.as_completed(tasks):
        try:
            results.append(await coro)
        except Exception as exc:
            errors += 1
            log.error("Profile generation failed: %s", exc)

    log.info("Profile generation complete: %d ok, %d errors", len(results), errors)
    return results
