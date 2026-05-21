"""
Comment generator: creates all three severity conditions for each post.

For every original post, three comments are generated:
  1. opposing_opinion   — severity level 1
  2. dehumanising       — severity level 2
  3. inciting_violence  — severity level 3

The post itself is fixed across conditions; only the comment varies.
This is the core experimental manipulation.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from src.llm_client import LLMClient
from src.models import Comment, CommentSeverity, OriginalPost, Profile
from src.policies import PolicyCondition
from src.prompts import PromptBuilder
from src.generators.posts import get_target_group
from src.sampling import Condition
from src.utils.seeds import derive_seed

log = logging.getLogger("pipeline.generators.comments")

SEVERITIES = [
    CommentSeverity.opposing_opinion,
    CommentSeverity.dehumanising,
    CommentSeverity.inciting_violence,
]


async def generate_comment(
    post: OriginalPost,
    profile: Profile,
    condition: Condition,
    severity: CommentSeverity,
    client: LLMClient,
    prompt_builder: PromptBuilder,
    base_seed: int,
    comment_index: int = 0,
) -> tuple[Comment, str]:
    """
    Generate and validate one Comment at the specified severity level.
    Returns (comment, prompt_hash).
    """
    comment_id = f"{post.post_id}_{severity.value.upper()[:3]}{comment_index:02d}"
    seed = derive_seed(base_seed, "comment", comment_id)
    raw_tg = get_target_group(condition.topic, condition.stance)   # str | None

    if isinstance(condition, PolicyCondition) and condition.opposing_stance:
        target_group = condition.target_group_override or raw_tg or ""
        system, user, prompt_hash = prompt_builder.build(
            f"comment_{severity.value}_policy",
            comment_id=comment_id,
            post_id=post.post_id,
            topic=post.topic.value,
            caption=post.caption,
            target_group=target_group,
            post_stance=condition.post_stance,
            opposing_stance=condition.opposing_stance,
        )
    else:
        target_group_context = (
            f"Target group implied by the post: {raw_tg}"
            if raw_tg
            else "Target group: infer from the post's content"
        )
        system, user, prompt_hash = prompt_builder.comment(
            severity=severity.value,
            comment_id=comment_id,
            post_id=post.post_id,
            topic=post.topic.value,
            caption=post.caption,
            target_group_context=target_group_context,
        )

    log.debug("Generating comment %s  severity=%s", comment_id, severity.value)
    raw = await client.complete_json(system, user, seed=seed)

    raw.setdefault("comment_id",               comment_id)
    raw.setdefault("post_id",                  post.post_id)
    raw.setdefault("severity",                 severity.value)
    raw.setdefault("target_group",             raw_tg or "")
    raw.setdefault("contains_explicit_violence",
                   severity == CommentSeverity.inciting_violence)

    # Clamp toxicity to valid range
    if "toxicity_estimate" in raw:
        raw["toxicity_estimate"] = float(
            max(0.0, min(1.0, raw["toxicity_estimate"]))
        )

    try:
        comment = Comment.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(
            f"Comment schema validation failed for {comment_id}: {exc}"
        ) from exc

    log.info("  ✓ Comment %s  [%s]  tox=%.2f",
             comment.comment_id, comment.severity.value, comment.toxicity_estimate)

    return comment, prompt_hash


async def generate_all_severity_comments(
    post: OriginalPost,
    profile: Profile,
    condition: Condition,
    client: LLMClient,
    prompt_builder: PromptBuilder,
    base_seed: int,
) -> list[tuple[Comment, str]]:
    """
    Generate all three severity comments for a single post.
    Comments are generated concurrently but all use the same post text.
    """
    import asyncio

    tasks = [
        asyncio.create_task(
            generate_comment(post, profile, condition, sev, client,
                             prompt_builder, base_seed, idx)
        )
        for idx, sev in enumerate(SEVERITIES)
    ]
    results: list[tuple[Comment, str]] = []
    for coro in asyncio.as_completed(tasks):
        try:
            results.append(await coro)
        except Exception as exc:
            log.error("Comment generation failed: %s", exc)

    return results


async def generate_comments_batch(
    post_profile_condition_triples: list[tuple[OriginalPost, Profile, Condition]],
    client: LLMClient,
    prompt_builder: PromptBuilder,
    base_seed: int,
) -> list[tuple[Comment, OriginalPost, Profile, Condition, str]]:
    """
    Generate all three severity comments for every post in the batch.
    Returns flat list of (comment, post, profile, condition, prompt_hash).
    """
    import asyncio

    all_results: list[tuple[Comment, OriginalPost, Profile, Condition, str]] = []
    errors = 0

    async def _one_post(
        post: OriginalPost, profile: Profile, cond: Condition
    ) -> list[tuple[Comment, OriginalPost, Profile, Condition, str]]:
        pairs = await generate_all_severity_comments(
            post, profile, cond, client, prompt_builder, base_seed
        )
        return [(c, post, profile, cond, ph) for c, ph in pairs]

    tasks = [
        asyncio.create_task(_one_post(p, pr, c))
        for p, pr, c in post_profile_condition_triples
    ]

    for coro in asyncio.as_completed(tasks):
        try:
            all_results.extend(await coro)
        except Exception as exc:
            errors += 1
            log.error("Batch comment generation failed: %s", exc)

    log.info("Comment generation complete: %d ok, %d errors", len(all_results), errors)
    return all_results
