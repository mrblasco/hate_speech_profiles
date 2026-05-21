"""
Generation pipeline: orchestrates the full stimulus generation workflow.

Stage order
-----------
  1. Load experimental design (from CSV or internal balanced sampler)
  2. Generate fictional Instagram profiles
  3. Generate non-hateful Instagram posts (one per unique profile × topic × stance)
  4. Validate posts (schema + rule + realism)
  5. Generate comments at the required severity levels (per post)
  6. Validate comments (schema + severity judge)
  7. Assemble StimulusRow objects
  8. Write outputs (CSV, JSONL, metadata JSON)

All stages are async to exploit concurrency within the LLM client's
semaphore budget. Intermediate artifacts are saved so a crash can be
diagnosed without rerunning the full pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.design_source import CsvDesignSource, DesignRow, InternalDesignSource
from src.generators.comments import generate_comments_batch, generate_comment
from src.generators.posts import generate_posts_batch, init_registry as init_posts_registry
from src.generators.profiles import generate_profiles_batch
from src.generators import html_stimuli
from src.llm_client import make_client
from src.models import (
    Comment, CommentSeverity, GenerationMeta, OriginalPost, Profile,
    RealismCheck, RunManifest, SeverityJudgement, StimulusRow,
)
from src.prompts import PromptBuilder
from src.sampling import Condition
from src.policies import PolicyCondition
from src.validators.realism import check_realism_batch
from src.validators.schema import SchemaValidator
from src.validators.severity import judge_comments_batch
from src.utils.hashing import sha256_file, sha256_text
from src.utils.io import DiskCache, TopicRegistry, append_jsonl, save_csv, save_json, save_jsonl
from src.utils.logging_utils import get_logger
from src.utils.seeds import derive_seed

log = get_logger("pipeline")

_ALL_SEVERITIES = list(CommentSeverity)


class GenerationConfig:
    """Parsed runtime configuration passed to the pipeline."""

    def __init__(
        self,
        study_cfg: dict,
        n_profiles: int,
        seed: int,
        output_dir: Path,
        model: str,
        enable_severity_judge: bool = True,
        enable_realism_check: bool = True,
        reject_on_mismatch: bool = True,
        dry_run: bool = False,
        generate_html: bool = False,
        screenshots: bool = False,
        policies_cfg: dict | None = None,
        csv_path: Path | None = None,
        country: str = "",
    ) -> None:
        self.study_cfg             = study_cfg
        self.n_profiles            = n_profiles
        self.seed                  = seed
        self.output_dir            = output_dir
        self.model                 = model
        self.enable_severity_judge = enable_severity_judge
        self.enable_realism_check  = enable_realism_check
        self.reject_on_mismatch    = reject_on_mismatch
        self.dry_run               = dry_run
        self.generate_html         = generate_html
        self.screenshots           = screenshots
        self.policies_cfg          = policies_cfg
        self.csv_path              = csv_path
        self.country               = country

        gen = study_cfg.get("generation", {})
        self.temperature         = gen.get("temperature", 0.9)
        self.max_tokens          = gen.get("max_tokens", 1024)
        self.max_retries         = gen.get("max_retries", 3)
        self.retry_delay         = gen.get("retry_delay_seconds", 5)
        self.rpm                 = gen.get("requests_per_minute", 30)
        self.concurrency         = gen.get("concurrency", 5)
        self.enable_cache        = gen.get("enable_cache", True)
        self.anonymous_display_name = gen.get("anonymous_display_name", "anonymous_user")

        design = study_cfg.get("design", {})
        self.topics              = design.get("topics",              [])
        self.age_groups          = design.get("age_groups",          [])
        self.genders             = design.get("genders",             [])
        self.stances             = design.get("stances",             ["support", "oppose"])
        self.religions           = design.get("religions",           [])
        self.countries_of_origin = design.get("countries_of_origin", [])
        self.severities = [s.value for s in CommentSeverity]


async def run_pipeline(
    cfg: GenerationConfig,
    configs_dir: Path,
) -> RunManifest:
    """
    Execute the full generation pipeline end-to-end.
    Returns the completed RunManifest.
    """
    experiment_id = sha256_text(
        f"{cfg.seed}|{cfg.n_profiles}|{cfg.model}"
    )[:12]
    run_id = uuid.uuid4().hex[:8]
    ts_start = datetime.utcnow()

    log.info("=" * 60)
    log.info("Experiment ID : %s", experiment_id)
    log.info("Run ID        : %s", run_id)
    log.info("Seed          : %d", cfg.seed)
    log.info("Model         : %s", cfg.model)
    log.info("Output dir    : %s", cfg.output_dir)
    log.info("=" * 60)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir  = cfg.output_dir / "raw"
    val_dir  = cfg.output_dir / "validated"
    fin_dir  = cfg.output_dir / "final"
    log_dir  = cfg.output_dir / "logs"
    for d in (raw_dir, val_dir, fin_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Shared components (lazy — not needed for dry-run) ─────────────────────
    prompt_builder   = PromptBuilder(configs_dir / "prompts.yaml")
    schema_validator = SchemaValidator(configs_dir / "generation_rules.yaml")

    topic_registry = TopicRegistry(configs_dir / "topics.yaml")
    init_posts_registry(topic_registry)
    html_stimuli.init_registry(topic_registry)

    # ── Stage 1: Load design ──────────────────────────────────────────────────
    csv_design: CsvDesignSource | None = None

    if cfg.csv_path:
        log.info("[Stage 1] Loading CSV design from %s …", cfg.csv_path)
        csv_design = CsvDesignSource(cfg.csv_path, topic_registry, country=cfg.country)
        conditions = csv_design.conditions()
        log.info("  %d unique profiles to generate", len(conditions))
    elif cfg.policies_cfg:
        from src.policies import build_policy_conditions, load_policies
        log.info("[Stage 1] Policy mode: building conditions from policies.yaml …")
        policies = [p for p in cfg.policies_cfg.get("policies", [])
                    if p.get("enabled", True)]
        conditions = build_policy_conditions(
            policies=policies,
            age_groups=cfg.age_groups,
            genders=cfg.genders,
            religions=cfg.religions,
            countries_of_origin=cfg.countries_of_origin,
            n_profiles=cfg.n_profiles,
            seed=cfg.seed,
        )
    else:
        log.info("[Stage 1] Sampling experimental conditions …")
        design_source = InternalDesignSource(cfg.study_cfg, cfg.n_profiles, cfg.seed)
        conditions = design_source.conditions()

    if cfg.dry_run:
        log.info("[dry-run] Would generate %d profiles. Exiting.", len(conditions))
        return _empty_manifest(experiment_id, run_id, ts_start, cfg)

    # ── LLM client (created after dry-run check to avoid import errors) ───────
    cache_dir = cfg.output_dir / ".cache" if cfg.enable_cache else None
    client = make_client(
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        max_retries=cfg.max_retries,
        retry_delay=cfg.retry_delay,
        requests_per_minute=cfg.rpm,
        cache_dir=cache_dir,
        enable_cache=cfg.enable_cache,
    )

    # ── Stage 2: Profile generation ───────────────────────────────────────────
    log.info("[Stage 2] Generating %d profiles …", len(conditions))
    profile_results = await generate_profiles_batch(
        conditions, client, prompt_builder, cfg.seed
    )
    profiles_data = [p.model_dump() for p, _, _ in profile_results]
    save_json(profiles_data, raw_dir / "profiles.json")

    # Build profile lookup: internal profile_id → Profile
    profile_map: dict[str, Profile] = {p.profile_id: p for p, _, _ in profile_results}

    # ── Stage 2b: Profile HTML pages (optional) ───────────────────────────────
    if cfg.generate_html:
        log.info("[Stage 2b] Rendering profile pages …")
        from src.generators.html_profiles import generate_profile_pages
        from src.generators.html_stimuli import take_screenshots
        project_root = Path(__file__).resolve().parents[2]
        profile_topic_pairs = [
            (p, c.topic if c.topic else "immigration")
            for p, c, _ in profile_results
        ]
        profile_html_paths = generate_profile_pages(
            profile_topic_pairs, cfg.output_dir / "final", project_root, cfg.seed
        )
        if cfg.screenshots:
            png_dir = cfg.output_dir / "final" / "png_profiles"
            await take_screenshots(profile_html_paths, png_dir)

    # ── Stage 3: Post generation ──────────────────────────────────────────────
    log.info("[Stage 3] Generating posts …")
    if csv_design:
        profile_cond_pairs = csv_design.unique_post_conditions(profile_map)
        log.info("  %d unique (profile, topic, stance) combinations", len(profile_cond_pairs))
    else:
        profile_cond_pairs = [(p, c) for p, c, _ in profile_results]

    post_results = await generate_posts_batch(
        profile_cond_pairs, client, prompt_builder, cfg.seed
    )
    posts_data = [p.model_dump() for p, _, _, _ in post_results]
    save_json(posts_data, raw_dir / "posts.json")

    # ── Stage 4: Post validation ──────────────────────────────────────────────
    log.info("[Stage 4] Validating posts …")
    valid_post_results: list[tuple[OriginalPost, Profile, Condition, str]] = []
    for post, profile, cond, ph in post_results:
        issues = schema_validator.validate_post(post)
        if issues:
            log.warning("Post %s failed schema validation: %s", post.post_id, issues)
            continue
        valid_post_results.append((post, profile, cond, ph))

    realism_map: dict[str, RealismCheck] = {}
    if cfg.enable_realism_check:
        log.info("[Stage 4b] Running realism checks …")
        realism_results = await check_realism_batch(
            [(p, pr, c) for p, pr, c, _ in valid_post_results],
            client, prompt_builder, cfg.seed,
        )
        realism_map = {p.post_id: rc for p, rc in realism_results}

    log.info("Posts: %d valid / %d generated", len(valid_post_results), len(post_results))

    # ── Stage 5: Comment generation ───────────────────────────────────────────
    log.info("[Stage 5] Generating comments …")

    if csv_design:
        # CSV mode: generate only the severities specified in the design rows
        comment_results = await _generate_csv_comments(
            valid_post_results, csv_design, client, prompt_builder, cfg.seed
        )
    else:
        # Standard mode: generate all 3 severities per post
        log.info("  (3 severities × %d posts)", len(valid_post_results))
        post_profile_cond_triples = [(p, pr, c) for p, pr, c, _ in valid_post_results]
        comment_results = await generate_comments_batch(
            post_profile_cond_triples, client, prompt_builder, cfg.seed
        )

    comments_data = [c.model_dump() for c, _, _, _, _ in comment_results]
    save_json(comments_data, raw_dir / "comments.json")

    # ── Stage 6: Comment validation ───────────────────────────────────────────
    log.info("[Stage 6] Validating comments …")
    valid_comment_results: list[tuple[Comment, OriginalPost, Profile, Condition, str]] = []
    for comment, post, profile, cond, ph in comment_results:
        issues = schema_validator.validate_comment(comment)
        if issues:
            log.warning("Comment %s failed schema validation: %s", comment.comment_id, issues)
            continue
        valid_comment_results.append((comment, post, profile, cond, ph))

    judge_map: dict[str, SeverityJudgement] = {}
    mismatch_ids: set[str] = set()
    if cfg.enable_severity_judge:
        log.info("[Stage 6b] Running severity judge …")
        comment_post_pairs = [(c, p) for c, p, _, _, _ in valid_comment_results]
        judge_results = await judge_comments_batch(
            comment_post_pairs, client, prompt_builder, cfg.seed
        )
        for comment, judgement, agrees in judge_results:
            judge_map[comment.comment_id] = judgement
            if not agrees and cfg.reject_on_mismatch:
                mismatch_ids.add(comment.comment_id)
                log.warning("Rejecting %s: judge=%s intended=%s",
                            comment.comment_id,
                            judgement.severity_label.value,
                            comment.severity.value)

    # ── Stage 7: Assemble stimulus rows ───────────────────────────────────────
    log.info("[Stage 7] Assembling stimulus rows …")
    stimulus_rows: list[StimulusRow] = []
    n_failed = 0

    if csv_design:
        stimulus_rows, n_failed = _assemble_csv_rows(
            valid_comment_results, judge_map, mismatch_ids,
            realism_map, cfg, run_id, experiment_id, csv_design,
        )
    else:
        for comment, post, profile, cond, ph in valid_comment_results:
            passed = comment.comment_id not in mismatch_ids
            if not passed:
                n_failed += 1
            meta = GenerationMeta(
                model_name=cfg.model,
                prompt_text=ph,
                temperature=cfg.temperature,
                seed=derive_seed(cfg.seed, "comment", comment.comment_id),
                run_id=run_id,
                experiment_id=experiment_id,
            )
            judge   = judge_map.get(comment.comment_id)
            realism = realism_map.get(post.post_id)
            row = StimulusRow.from_parts(
                profile=profile, post=post, comment=comment, meta=meta,
                judge=judge, realism=realism, passed=passed,
                policy_id=cond.policy_id if isinstance(cond, PolicyCondition) else None,
                post_stance=cond.post_stance if isinstance(cond, PolicyCondition) else None,
                opposing_stance=cond.opposing_stance if isinstance(cond, PolicyCondition) else None,
            )
            stimulus_rows.append(row)

    # ── Stage 8: Write outputs ────────────────────────────────────────────────
    log.info("[Stage 8] Writing outputs …")
    rows_dict   = [r.model_dump() for r in stimulus_rows]
    rows_passed = [r for r in rows_dict if r["passed_validation"]]

    csv_path   = fin_dir / "stimuli.csv"
    jsonl_path = fin_dir / "stimuli.jsonl"
    meta_path  = fin_dir / "generation_metadata.json"
    val_path   = val_dir / "stimuli_all.jsonl"

    save_csv(rows_passed, csv_path)
    save_jsonl(rows_passed, jsonl_path)
    save_jsonl(rows_dict, val_path)

    ts_end      = datetime.utcnow()
    config_hash = sha256_file(configs_dir / "study_config.yaml")

    manifest = RunManifest(
        experiment_id=experiment_id,
        run_id=run_id,
        timestamp_start=ts_start,
        timestamp_end=ts_end,
        seed=cfg.seed,
        n_profiles=len(profile_results),
        n_posts=len(valid_post_results),
        n_comments=len(valid_comment_results),
        n_stimuli=len(stimulus_rows),
        n_passed=len(rows_passed),
        n_failed=n_failed,
        model_name=cfg.model,
        temperature=cfg.temperature,
        config_hash=config_hash,
        topics=cfg.topics,
        age_groups=cfg.age_groups,
        genders=cfg.genders,
        stances=cfg.stances,
        severities=cfg.severities,
        output_files={
            "csv":      str(csv_path),
            "jsonl":    str(jsonl_path),
            "metadata": str(meta_path),
            "all":      str(val_path),
        },
    )
    save_json(manifest.model_dump(), meta_path)

    # ── Stage 9: HTML (+ optional PNG) rendering ──────────────────────────────
    if cfg.generate_html:
        log.info("[Stage 9] Rendering HTML stimuli …")
        from src.generators.html_stimuli import generate_html_stimuli, take_screenshots

        project_root      = Path(__file__).resolve().parents[2]
        passed_row_objects = [StimulusRow.model_validate(r) for r in rows_passed]

        if csv_design:
            # Render each row with its assigned engagement level only (3 variants)
            for row in passed_row_objects:
                likes_level = row.popularity or "active user"
                likes_filter = (
                    "low"  if likes_level == "ordinary user"  else
                    "mid"  if likes_level == "active user"    else
                    "high"
                )
                html_paths = generate_html_stimuli(
                    [row], fin_dir, project_root, cfg.seed,
                    likes_filter=likes_filter,
                    anonymous_display_name=cfg.anonymous_display_name,
                )
                if cfg.screenshots:
                    await take_screenshots(html_paths, fin_dir / "png")
        else:
            html_paths = generate_html_stimuli(
                passed_row_objects, fin_dir, project_root, cfg.seed,
                anonymous_display_name=cfg.anonymous_display_name,
            )
            manifest.output_files["html_dir"] = str(fin_dir / "html")
            if cfg.screenshots:
                png_dir = fin_dir / "png"
                await take_screenshots(html_paths, png_dir)
                manifest.output_files["png_dir"] = str(png_dir)

        save_json(manifest.model_dump(), meta_path)

    log.info("=" * 60)
    log.info("Run complete.")
    log.info("  Profiles   : %d", manifest.n_profiles)
    log.info("  Posts      : %d", manifest.n_posts)
    log.info("  Comments   : %d", manifest.n_comments)
    log.info("  Stimuli    : %d  (%d passed, %d failed)",
             manifest.n_stimuli, manifest.n_passed, manifest.n_failed)
    log.info("  Cache      : %s", client.cache_stats)
    log.info("  Outputs    → %s", fin_dir)
    log.info("=" * 60)

    return manifest


# ── CSV-specific helpers ──────────────────────────────────────────────────────

async def _generate_csv_comments(
    valid_post_results: list[tuple[OriginalPost, Profile, Condition, str]],
    csv_design: CsvDesignSource,
    client,
    prompt_builder: PromptBuilder,
    base_seed: int,
) -> list[tuple[Comment, OriginalPost, Profile, Condition, str]]:
    """
    For each valid post, generate only the severity levels present in the
    CSV design rows (not all 3).
    """
    import asyncio

    # Key by (profile_id, topic, stance) — avoids reconstructing post_ids,
    # which are enumeration-indexed and don't encode topic/stance.
    post_cond_map: dict[tuple[str, str, str], tuple] = {}
    for post, profile, cond, _ in valid_post_results:
        key = (cond.profile_id, cond.topic, cond.stance)
        post_cond_map[key] = (post, profile, cond)

    # Determine unique (profile_id, topic, stance, severity) combos from design
    seen: set[tuple[str, str, str, CommentSeverity]] = set()
    tasks_to_run: list[tuple[OriginalPost, Profile, Condition, CommentSeverity]] = []

    for row in csv_design.rows():
        internal_pid = f"CSV_{row.profile_id:0>4}"
        lookup_key = (internal_pid, row.topic, row.stance)
        task_key   = (internal_pid, row.topic, row.stance, row.severity)
        if task_key not in seen:
            if lookup_key in post_cond_map:
                seen.add(task_key)
                post, profile, cond = post_cond_map[lookup_key]
                tasks_to_run.append((post, profile, cond, row.severity))
            else:
                log.warning(
                    "Post not found for profile=%s topic=%s stance=%s — no comment will be generated",
                    internal_pid, row.topic, row.stance,
                )

    log.info("  %d unique (post, severity) pairs to generate", len(tasks_to_run))

    all_results: list[tuple[Comment, OriginalPost, Profile, Condition, str]] = []
    errors = 0

    async def _one(post, profile, cond, sev, idx):
        comment, ph = await generate_comment(
            post, profile, cond, sev, client, prompt_builder, base_seed,
            comment_index=idx,
        )
        return comment, post, profile, cond, ph

    tasks = [
        asyncio.create_task(_one(p, pr, c, sev, i))
        for i, (p, pr, c, sev) in enumerate(tasks_to_run)
    ]
    for coro in asyncio.as_completed(tasks):
        try:
            all_results.append(await coro)
        except Exception as exc:
            errors += 1
            log.error("CSV comment generation failed: %s", exc)

    log.info("CSV comment generation complete: %d ok, %d errors", len(all_results), errors)
    return all_results


def _assemble_csv_rows(
    valid_comment_results: list[tuple[Comment, OriginalPost, Profile, Condition, str]],
    judge_map: dict[str, SeverityJudgement],
    mismatch_ids: set[str],
    realism_map: dict[str, RealismCheck],
    cfg: GenerationConfig,
    run_id: str,
    experiment_id: str,
    csv_design: CsvDesignSource,
) -> tuple[list[StimulusRow], int]:
    """
    Assemble one StimulusRow per design row, enriched with respondent_id,
    anonymity, and popularity for downstream R join-back.
    """
    # Key: (profile_id, topic, stance, severity) — mirrors _generate_csv_comments.
    # We avoid reconstructing post_ids because they are enumeration-indexed.
    comment_index: dict[tuple[str, str, str, CommentSeverity], tuple] = {}
    for comment, post, profile, cond, ph in valid_comment_results:
        key = (cond.profile_id, cond.topic, cond.stance, comment.severity)
        comment_index[key] = (comment, post, profile, cond, ph)

    stimulus_rows: list[StimulusRow] = []
    n_failed = 0

    for design_row in csv_design.rows():
        internal_pid = f"CSV_{design_row.profile_id:0>4}"
        key = (internal_pid, design_row.topic, design_row.stance, design_row.severity)

        if key not in comment_index:
            log.warning(
                "No generated comment for respondent=%s profile=%s topic=%s stance=%s severity=%s — skipping row",
                design_row.respondent_id, internal_pid, design_row.topic,
                design_row.stance, design_row.severity.value,
            )
            continue

        comment, post, profile, cond, ph = comment_index[key]
        passed = comment.comment_id not in mismatch_ids
        if not passed:
            n_failed += 1

        meta = GenerationMeta(
            model_name=cfg.model,
            prompt_text=ph,
            temperature=cfg.temperature,
            seed=derive_seed(cfg.seed, "comment", comment.comment_id),
            run_id=run_id,
            experiment_id=experiment_id,
        )
        judge   = judge_map.get(comment.comment_id)
        realism = realism_map.get(post.post_id)

        row = StimulusRow.from_parts(
            profile=profile, post=post, comment=comment, meta=meta,
            judge=judge, realism=realism, passed=passed,
            respondent_id=design_row.respondent_id,
            anonymity=design_row.anonymity,
            popularity=design_row.popularity,
        )
        stimulus_rows.append(row)

    return stimulus_rows, n_failed


def _empty_manifest(
    experiment_id: str,
    run_id: str,
    ts_start: datetime,
    cfg: "GenerationConfig",
) -> RunManifest:
    return RunManifest(
        experiment_id=experiment_id,
        run_id=run_id,
        timestamp_start=ts_start,
        timestamp_end=datetime.utcnow(),
        seed=cfg.seed,
        n_profiles=0, n_posts=0, n_comments=0,
        n_stimuli=0, n_passed=0, n_failed=0,
        model_name=cfg.model,
        temperature=cfg.temperature,
        config_hash="",
        topics=cfg.topics, age_groups=cfg.age_groups,
        genders=cfg.genders, stances=cfg.stances,
        severities=cfg.severities,
    )
