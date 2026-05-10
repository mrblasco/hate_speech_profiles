"""
Profile generator: creates fictional Instagram profiles via LLM.

Each profile is generated independently with a derived seed so results are
reproducible and auditable. Profiles are validated against the Profile schema
before being returned.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from src.llm_client import LLMClient
from src.models import AgeGroup, Profile, Religion, CountryOfOrigin
from src.prompts import PromptBuilder
from src.sampling import Condition
from src.utils.seeds import derive_seed

log = logging.getLogger("pipeline.generators.profiles")

# Age ranges by group (must match study_config.yaml)
AGE_RANGES: dict[str, tuple[int, int]] = {
    "adolescent":  (13, 17),
    "young_adult": (18, 25),
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
    age_min, age_max = AGE_RANGES[condition.age_group]
    seed = derive_seed(base_seed, "profile", condition.profile_id)

    system, user, prompt_hash = prompt_builder.profile(
        profile_id=condition.profile_id,
        topic=condition.topic,
        age_group=condition.age_group,
        age_min=age_min,
        age_max=age_max,
        gender=condition.gender,
        values=condition.values,
        religion=condition.religion,
        country_of_origin=condition.country_of_origin,
    )

    log.debug("Generating profile %s (topic=%s, %s/%s/%s)",
              condition.profile_id, condition.topic,
              condition.age_group, condition.gender, condition.values)

    raw = await client.complete_json(system, user, seed=seed)

    # Ensure required fields that the LLM might omit
    raw.setdefault("profile_id",        condition.profile_id)
    raw.setdefault("age_group",         condition.age_group)
    raw.setdefault("gender",            condition.gender)
    raw.setdefault("values",            condition.values)
    raw.setdefault("religion",          condition.religion)
    raw.setdefault("country_of_origin", condition.country_of_origin)

    try:
        profile = Profile.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(
            f"Profile schema validation failed for {condition.profile_id}: {exc}"
        ) from exc

    log.info("  ✓ Profile %s  @%s  (%s, %s, %s)",
             profile.profile_id, profile.username,
             profile.topic if hasattr(profile, "topic") else condition.topic,
             profile.gender.value, profile.values.value)

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
