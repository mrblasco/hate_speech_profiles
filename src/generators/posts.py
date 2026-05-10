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
from src.models import OriginalPost, Profile, Topic
from src.prompts import PromptBuilder
from src.sampling import Condition
from src.utils.seeds import derive_seed

log = logging.getLogger("pipeline.generators.posts")

# Maps topic enum values to human-readable labels used in prompts
TOPIC_LABELS: dict[str, str] = {
    "immigration":       "immigration and migration policy",
    "feminism":          "feminism and gender equality",
    "religion":          "religion and religious freedom",
    "climate":           "climate change and environmental policy",
    "public_health":     "public health policy and vaccines",
    "national_identity": "national identity and patriotism",
}

# Maps topic + values to implied target group (used for comment generation)
TARGET_GROUPS: dict[tuple[str, str], str] = {
    ("immigration",       "progressive"):  "immigrants and refugees",
    ("immigration",       "conservative"): "immigration critics",
    ("feminism",          "progressive"):  "women and feminist activists",
    ("feminism",          "conservative"): "traditionalist women and men",
    ("religion",          "progressive"):  "religious minorities",
    ("religion",          "conservative"): "secularists",
    ("climate",           "progressive"):  "climate activists",
    ("climate",           "conservative"): "climate-change sceptics",
    ("public_health",     "progressive"):  "public health advocates",
    ("public_health",     "conservative"): "vaccine sceptics",
    ("national_identity", "progressive"):  "multicultural communities",
    ("national_identity", "conservative"): "nationalists",
}


def get_target_group(topic: str, values: str) -> str:
    return TARGET_GROUPS.get((topic, values), "the poster's community")


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
    seed = derive_seed(base_seed, "post", post_id)
    topic_label = TOPIC_LABELS.get(condition.topic, condition.topic)

    system, user, prompt_hash = prompt_builder.post(
        post_id=post_id,
        profile_id=profile.profile_id,
        username=profile.username,
        age_group=condition.age_group,
        gender=condition.gender,
        values=condition.values,
        writing_style=profile.writing_style,
        topic=topic_label,
    )

    log.debug("Generating post %s", post_id)
    raw = await client.complete_json(system, user, seed=seed)

    raw.setdefault("post_id",    post_id)
    raw.setdefault("profile_id", profile.profile_id)
    raw.setdefault("topic",      condition.topic)

    # Normalise topic string to enum value
    topic_val = raw.get("topic", condition.topic)
    if topic_val not in [t.value for t in Topic]:
        raw["topic"] = condition.topic

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
