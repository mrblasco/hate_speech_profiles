"""
Post generator: creates non-hateful Instagram captions for each profile.

The caption expresses an opinionated but hate-speech-free perspective on the
profile's topic. Word count and style are constrained by prompts and validated
post-generation.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from src.llm_client import LLMClient
from src.models import OriginalPost, Profile
from src.prompts import PromptBuilder
from src.sampling import Condition
from src.utils.seeds import derive_seed

log = logging.getLogger("pipeline.generators.posts")


def get_target_group(topic: str) -> str:
    """Infer the implied target group from the topic stance."""
    t = topic.lower()
    if t.startswith("supports "):
        return "supporters of " + topic[len("supports "):]
    if t.startswith("opposes "):
        return "critics of " + topic[len("opposes "):]
    return "the poster's community"


async def generate_post(
    profile: Profile,
    condition: Condition,
    client: LLMClient,
    prompt_builder: PromptBuilder,
    base_seed: int,
    post_index: int = 0,
) -> tuple[OriginalPost, str]:
    """
    Generate and validate one OriginalPost for the given profile.
    Returns (post, prompt_hash).
    """
    post_id = f"{profile.profile_id}_POST{post_index:02d}"
    seed    = derive_seed(base_seed, "post", post_id)
    topic   = condition.factors.get("post_topic", "")
    values  = "progressive" if topic.startswith("supports") else "conservative"

    system, user, prompt_hash = prompt_builder.post(
        post_id=post_id,
        profile_id=profile.profile_id,
        username=profile.username,
        age_group=condition.factors.get("target_age_group", ""),
        gender=condition.factors.get("target_gender", ""),
        values=values,
        writing_style=profile.writing_style,
        topic=topic,
        respondent_country=condition.factors.get("respondent_country", ""),
    )

    log.debug("Generating post %s", post_id)
    raw = await client.complete_json(system, user, seed=seed)

    raw.setdefault("post_id",    post_id)
    raw.setdefault("profile_id", profile.profile_id)
    raw.setdefault("topic",      topic)

    try:
        post = OriginalPost.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Post schema validation failed for {post_id}: {exc}") from exc

    log.info("  ✓ Post %s  (%d words)  %s", post.post_id, post.word_count,
             post.caption[:60] + ("…" if len(post.caption) > 60 else ""))

    return post, prompt_hash


async def generate_posts_batch(
    profile_condition_pairs: list[tuple[Profile, Condition]],
    client: LLMClient,
    prompt_builder: PromptBuilder,
    base_seed: int,
) -> list[tuple[OriginalPost, Profile, Condition, str]]:
    """Generate posts for all (profile, condition) pairs concurrently."""
    import asyncio

    async def _one(
        profile: Profile, cond: Condition, idx: int
    ) -> tuple[OriginalPost, Profile, Condition, str]:
        post, ph = await generate_post(profile, cond, client, prompt_builder, base_seed, idx)
        return post, profile, cond, ph

    tasks = [asyncio.create_task(_one(p, c, i)) for i, (p, c) in enumerate(profile_condition_pairs)]
    results: list[tuple[OriginalPost, Profile, Condition, str]] = []
    errors = 0

    for coro in asyncio.as_completed(tasks):
        try:
            results.append(await coro)
        except Exception as exc:
            errors += 1
            log.error("Post generation failed: %s", exc)

    log.info("Post generation complete: %d ok, %d errors", len(results), errors)
    return results
