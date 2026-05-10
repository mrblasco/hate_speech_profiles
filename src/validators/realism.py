"""
Realism validator: uses an LLM judge to assess whether generated posts
sound like authentic social media content.

Applied to original posts only (not comments) because comment realism is
a secondary concern compared to severity calibration.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from src.llm_client import LLMClient
from src.models import OriginalPost, Profile, RealismCheck
from src.prompts import PromptBuilder
from src.sampling import Condition
from src.utils.seeds import derive_seed

log = logging.getLogger("pipeline.validators.realism")


async def check_post_realism(
    post: OriginalPost,
    profile: Profile,
    condition: Condition,
    client: LLMClient,
    prompt_builder: PromptBuilder,
    base_seed: int,
) -> RealismCheck:
    """
    Ask the LLM whether the post caption sounds realistic.
    Returns a RealismCheck with is_realistic, realism_score, and any issues.
    """
    seed = derive_seed(base_seed, "realism", post.post_id)

    system, user, _ = prompt_builder.realism_check(
        age_group=condition.age_group,
        gender=condition.gender,
        values=condition.values,
        topic=post.topic.value,
        caption=post.caption,
    )

    raw = await client.complete_json(system, user, seed=seed)
    raw.setdefault("issues", [])

    try:
        check = RealismCheck.model_validate(raw)
    except ValidationError as exc:
        log.warning("RealismCheck parse error for %s: %s", post.post_id, exc)
        check = RealismCheck(is_realistic=True, realism_score=0.5, issues=["parse error"])

    if not check.is_realistic:
        log.warning("Post %s flagged as unrealistic (score=%.2f): %s",
                    post.post_id, check.realism_score, "; ".join(check.issues))
    return check


async def check_realism_batch(
    post_profile_condition_triples: list[tuple[OriginalPost, Profile, Condition]],
    client: LLMClient,
    prompt_builder: PromptBuilder,
    base_seed: int,
) -> list[tuple[OriginalPost, RealismCheck]]:
    """Check realism for all posts concurrently."""
    import asyncio

    async def _one(
        post: OriginalPost, profile: Profile, cond: Condition
    ) -> tuple[OriginalPost, RealismCheck]:
        rc = await check_post_realism(post, profile, cond, client, prompt_builder, base_seed)
        return post, rc

    tasks = [asyncio.create_task(_one(p, pr, c)) for p, pr, c in post_profile_condition_triples]
    results: list[tuple[OriginalPost, RealismCheck]] = []

    for coro in asyncio.as_completed(tasks):
        try:
            results.append(await coro)
        except Exception as exc:
            log.error("Realism check failed: %s", exc)

    realistic = sum(1 for _, rc in results if rc.is_realistic)
    log.info("Realism check: %d/%d posts realistic", realistic, len(results))
    return results
