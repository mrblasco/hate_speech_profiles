"""
Generation pipeline: orchestrates the full stimulus generation workflow.

Stage order
-----------
  1. Sample experimental conditions (balanced factorial design)
  2. Generate fictional Instagram profiles
  3. Generate non-hateful Instagram posts (one per profile)
  4. Validate posts (schema + rule + realism)
  5. Generate comments at all 3 severity levels (per post)
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

from src.generators.comments import generate_comments_batch
from src.generators.posts import generate_posts_batch
from src.generators.profiles import generate_profiles_batch
from src.llm_client import make_client
from src.models import (
    Comment, GenerationMeta, OriginalPost, Profile,
    RealismCheck, RunManifest, SeverityJudgement, StimulusRow,
)
from src.prompts import PromptBuilder
from src.sampling import Condition, build_design_matrix
from src.validators.realism import check_realism_batch
from src.validators.schema import SchemaValidator
from src.validators.severity import judge_comments_batch
from src.utils.hashing import sha256_file, sha256_text
from src.utils.io import DiskCache, append_jsonl, save_csv, save_json, save_jsonl
from src.utils.logging_utils import get_logger
from src.utils.seeds import derive_seed

log = get_logger("pipeline")


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
    ) -> None:
        self.study_cfg           = study_cfg
        self.n_profiles          = n_profiles
        self.seed                = seed
        self.output_dir          = output_dir
        self.model               = model
        self.enable_severity_judge = enable_severity_judge
        self.enable_realism_check  = enable_realism_check
        self.reject_on_mismatch    = reject_on_mismatch
        self.dry_run             = dry_run
        self.generate_html       = generate_html
        self.screenshots         = screenshots

        gen = study_cfg.get("generation", {})
        self.temperature         = gen.get("temperature", 0.9)
        self.max_tokens          = gen.get("max_tokens", 1024)
        self.max_retries         = gen.get("max_retries", 3)
        self.retry_delay         = gen.get("retry_delay_seconds", 5)
        self.rpm                 = gen.get("requests_per_minute", 30)
        self.concurrency         = gen.get("concurrency", 5)
        self.enable_cache        = gen.get("enable_cache", True)

        design    = study_cfg.get("design", {})
        enum_keys = study_cfg.get("enums",  {})

        # comment_severities drives LLM generation; stimulus_factors are display-only.
        # Neither belongs in the between-profile design matrix.
        _stimulus_keys = set(study_cfg.get("stimulus_factors", []))
        _not_factors   = {"comment_severities"} | _stimulus_keys
        self.design_factors: dict[str, list[str]] = {
            key: design.get(key, [])
            for key in enum_keys
            if key not in _not_factors and isinstance(design.get(key), list)
        }
        self.comment_severities: list[str]        = design.get("comment_severities", [])
        self.stimulus_factors:   dict[str, list[str]] = {
            k: design.get(k, []) for k in _stimulus_keys
        }
        self.age_ranges: dict = design.get("age_ranges", {})


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
    log.info("Profiles      : %d", cfg.n_profiles)
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

    # ── Instantiate shared components ─────────────────────────────────────────
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

    prompt_builder = PromptBuilder(configs_dir / "prompts.yaml")
    schema_validator = SchemaValidator(configs_dir / "generation_rules.yaml")

    # ── Stage 1: Design matrix ────────────────────────────────────────────────
    log.info("[Stage 1] Sampling experimental conditions …")
    conditions = build_design_matrix(
        n_profiles=cfg.n_profiles,
        factors=cfg.design_factors,
        seed=cfg.seed,
    )

    if cfg.dry_run:
        log.info("[dry-run] Would generate %d profiles. Exiting.", len(conditions))
        return _empty_manifest(experiment_id, run_id, ts_start, cfg)

    # ── Stage 2: Profile generation ───────────────────────────────────────────
    log.info("[Stage 2] Generating %d profiles …", len(conditions))
    profile_results = await generate_profiles_batch(
        conditions, client, prompt_builder, cfg.seed
    )
    profiles_data   = [p.model_dump() for p, _, _ in profile_results]
    save_json(profiles_data, raw_dir / "profiles.json")

    # ── Stage 2b: Profile HTML pages (optional) ───────────────────────────────
    if cfg.generate_html:
        log.info("[Stage 2b] Rendering profile pages …")
        from src.generators.html_profiles import generate_profile_pages
        from src.generators.html_stimuli import take_screenshots
        project_root = Path(__file__).resolve().parents[2]
        profile_topic_pairs = [(p, c.factors.get("post_topic", "")) for p, c, _ in profile_results]
        profile_html_paths = generate_profile_pages(
            profile_topic_pairs, cfg.output_dir / "final", project_root, cfg.seed
        )
        if cfg.screenshots:
            png_dir = cfg.output_dir / "final" / "png_profiles"
            await take_screenshots(profile_html_paths, png_dir)

    # ── Stage 3: Post generation ──────────────────────────────────────────────
    log.info("[Stage 3] Generating posts …")
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
    log.info("[Stage 5] Generating comments (3 severities × %d posts) …",
             len(valid_post_results))
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

    for comment, post, profile, cond, ph in valid_comment_results:
        passed = comment.comment_id not in mismatch_ids
        if not passed:
            n_failed += 1

        meta = GenerationMeta(
            model_name=cfg.model,
            prompt_text=ph,   # store hash as text proxy for brevity
            temperature=cfg.temperature,
            seed=derive_seed(cfg.seed, "comment", comment.comment_id),
            run_id=run_id,
            experiment_id=experiment_id,
        )

        judge = judge_map.get(comment.comment_id)
        realism = realism_map.get(post.post_id)

        row = StimulusRow.from_parts(
            profile=profile,
            post=post,
            comment=comment,
            meta=meta,
            judge=judge,
            realism=realism,
            passed=passed,
        )
        stimulus_rows.append(row)

    # ── Stage 8: Write outputs ────────────────────────────────────────────────
    log.info("[Stage 8] Writing outputs …")
    rows_dict = [r.model_dump() for r in stimulus_rows]
    rows_passed = [r for r in rows_dict if r["passed_validation"]]

    csv_path    = fin_dir / "stimuli.csv"
    jsonl_path  = fin_dir / "stimuli.jsonl"
    meta_path   = fin_dir / "generation_metadata.json"
    val_path    = val_dir / "stimuli_all.jsonl"

    save_csv(rows_passed, csv_path)
    save_jsonl(rows_passed, jsonl_path)
    save_jsonl(rows_dict, val_path)    # includes failed rows for audit

    ts_end = datetime.utcnow()
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
        design_factors=cfg.design_factors,
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

        project_root = Path(__file__).resolve().parents[2]
        passed_row_objects = [StimulusRow.model_validate(r) for r in rows_passed]
        html_paths = generate_html_stimuli(
            passed_row_objects, fin_dir, project_root, cfg.seed
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
        design_factors=cfg.design_factors,
    )
