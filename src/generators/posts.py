"""
Post generator: creates non-hateful Instagram captions for each profile.

The caption expresses an opinionated but hate-speech-free perspective on the
profile's topic. Word count and style are constrained by prompts and validated
post-generation.

Topic labels and target groups are read from configs/topics.yaml via
TopicRegistry — no hardcoded dicts in this file.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from src.llm_client import LLMClient
from src.models import OriginalPost, Profile, Topic
from src.policies import PolicyCondition
from src.prompts import PromptBuilder
from src.sampling import Condition
from src.utils.io import TopicRegistry
from src.utils.seeds import derive_seed

log = logging.getLogger("pipeline.generators.posts")

_registry: TopicRegistry | None = None


def init_registry(registry: TopicRegistry) -> None:
    """Call once at pipeline startup to inject the shared TopicRegistry."""
    global _registry
    _registry = registry


def _get_registry() -> TopicRegistry:
    if _registry is None:
        raise RuntimeError(
            "TopicRegistry not initialised. Call posts.init_registry() first."
        )
    return _registry


def get_target_group(topic: str, stance: str) -> str:
    return _get_registry().target_group(topic, stance)


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
    registry = _get_registry()
    topic_label   = registry.prompt_label(condition.topic)  if condition.topic else condition.topic
    stance_label  = registry.stance_label(condition.topic, condition.stance) \
                    if condition.topic else condition.stance

    if isinstance(condition, PolicyCondition) and condition.post_stance:
        system, user, prompt_hash = prompt_builder.build(
            "post_generation_policy",
            post_id=post_id,
            profile_id=profile.profile_id,
            username=profile.username,
            age_group=condition.age_group,
            gender=condition.gender,
            stance=condition.stance,
            writing_style=profile.writing_style,
            topic=topic_label,
            post_stance=condition.post_stance,
        )
    else:
        system, user, prompt_hash = prompt_builder.post(
            post_id=post_id,
            profile_id=profile.profile_id,
            username=profile.username,
            age_group=condition.age_group or "young_adult",
            gender=condition.gender,
            stance=condition.stance,
            stance_label=stance_label,
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
