"""
LLM-based severity judge.

A second LLM call independently classifies each generated comment into one
of the three severity levels. Mismatches between intended and judged severity
are flagged and can trigger rejection.

This replication check increases construct validity — it ensures the generated
content actually operationalises the intended experimental condition.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from src.llm_client import LLMClient
from src.models import Comment, CommentSeverity, OriginalPost, SeverityJudgement
from src.prompts import PromptBuilder
from src.utils.seeds import derive_seed

log = logging.getLogger("pipeline.validators.severity")

_SEVERITY_TO_SCORE = {m: i + 1 for i, m in enumerate(CommentSeverity)}


async def judge_severity(
    comment: Comment,
    post: OriginalPost,
    client: LLMClient,
    prompt_builder: PromptBuilder,
    base_seed: int,
) -> SeverityJudgement:
    """
    Ask the LLM to independently classify the comment severity.
    Returns a SeverityJudgement with score, label, confidence, and reasoning.
    """
    seed = derive_seed(base_seed, "judge", comment.comment_id)

    system, user, _ = prompt_builder.severity_judge(
        text=comment.text,
        topic=post.topic,
    )

    raw = await client.complete_json(system, user, seed=seed)

    # Normalise severity_label if the LLM returned an integer score only
    if "severity_score" in raw and "severity_label" not in raw:
        score_to_label = {v: k.value for k, v in _SEVERITY_TO_SCORE.items()}
        raw["severity_label"] = score_to_label.get(int(raw["severity_score"]), "opposing_opinion")

    raw.setdefault("confidence", 0.5)
    raw.setdefault("reasoning", "")

    try:
        return SeverityJudgement.model_validate(raw)
    except ValidationError as exc:
        log.warning("SeverityJudgement parse error for %s: %s", comment.comment_id, exc)
        # Return a low-confidence judgement rather than crashing the pipeline
        return SeverityJudgement(
            severity_score=raw.get("severity_score", 1),
            severity_label=next(iter(CommentSeverity)),
            confidence=0.0,
            reasoning="parse error",
        )


def check_agreement(
    comment: Comment,
    judgement: SeverityJudgement,
) -> bool:
    """Return True if the judge's label matches the intended severity."""
    return judgement.severity_label == comment.severity


async def judge_comments_batch(
    comment_post_pairs: list[tuple[Comment, OriginalPost]],
    client: LLMClient,
    prompt_builder: PromptBuilder,
    base_seed: int,
) -> list[tuple[Comment, SeverityJudgement, bool]]:
    """
    Judge all comments concurrently.
    Returns list of (comment, judgement, agrees).
    """
    import asyncio

    async def _one(
        comment: Comment, post: OriginalPost
    ) -> tuple[Comment, SeverityJudgement, bool]:
        j = await judge_severity(comment, post, client, prompt_builder, base_seed)
        agrees = check_agreement(comment, j)
        if not agrees:
            log.warning(
                "Severity mismatch: intended=%s  judged=%s  conf=%.2f  [%s]",
                comment.severity.value, j.severity_label.value, j.confidence,
                comment.text[:60],
            )
        return comment, j, agrees

    tasks = [asyncio.create_task(_one(c, p)) for c, p in comment_post_pairs]
    results: list[tuple[Comment, SeverityJudgement, bool]] = []

    for coro in asyncio.as_completed(tasks):
        try:
            results.append(await coro)
        except Exception as exc:
            log.error("Severity judge failed: %s", exc)

    agreed = sum(1 for _, _, a in results if a)
    total = len(results)
    if total:
        log.info("Judge agreement: %d/%d (%.1f%%)", agreed, total, 100 * agreed / total)

    return results
