#!/usr/bin/env python3
"""
Instagram Vignette Stimulus Generator
======================================
CLI entry point for the LLM-based experimental stimulus generation pipeline.

Usage
-----
    python src/main.py --n_profiles 50 --seed 42 --output_dir outputs/run_001

    # Anthropic Claude model:
    python src/main.py --n_profiles 50 --model claude-sonnet-4-6

    # Dry run (design matrix only, no API calls):
    python src/main.py --n_profiles 10 --dry-run

    # Skip validation steps (faster, less rigorous):
    python src/main.py --n_profiles 50 --no-judge --no-realism

Environment variables (set in .env):
    OPENAI_API_KEY        — required for OpenAI / compatible models
    OPENAI_BASE_URL       — optional, for alternate OpenAI-compatible providers
    OPENAI_MODEL          — default model when ANTHROPIC_API_KEY is not set
    ANTHROPIC_API_KEY     — required for claude-* models
    ANTHROPIC_MODEL       — default model when ANTHROPIC_API_KEY is set
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root on sys.path so `src.*` imports work
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipelines.generation_pipeline import GenerationConfig, run_pipeline
from src.utils.io import load_yaml
from src.utils.logging_utils import setup_logging


CONFIGS_DIR = _ROOT / "configs"

# If ANTHROPIC_API_KEY is present, default to a Claude model; otherwise OpenAI.
DEFAULT_MODEL = (
    os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    if os.getenv("ANTHROPIC_API_KEY")
    else os.getenv("OPENAI_MODEL", "gpt-4o")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Instagram vignette stimuli for hate speech research.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n_profiles", type=int, default=50,
        help="Number of profiles (and posts) to generate. Default: 50.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master random seed for reproducibility. Default: 42.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("outputs/run_001"),
        help="Directory for all outputs. Default: outputs/run_001.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"LLM model name. Default: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip LLM severity judge (faster, less rigorous).",
    )
    parser.add_argument(
        "--no-realism", action="store_true",
        help="Skip realism check on posts.",
    )
    parser.add_argument(
        "--no-reject", action="store_true",
        help="Keep items even when judge disagrees with intended severity.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print design matrix only; make no API calls.",
    )
    parser.add_argument(
        "--from-jsonl", type=Path, default=None, metavar="PATH",
        help="Render HTML from an existing stimuli.jsonl; skips all LLM generation. "
             "--output-dir defaults to the JSONL file's parent directory.",
    )
    parser.add_argument(
        "--html", action="store_true",
        help="Render HTML stimuli via templates/ after generation.",
    )
    parser.add_argument(
        "--screenshots", action="store_true",
        help="Take Playwright PNG screenshots (implies --html; requires playwright).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIGS_DIR / "study_config.yaml",
        help="Path to study_config.yaml. Default: configs/study_config.yaml.",
    )
    parser.add_argument(
        "--policies", type=Path, default=None, metavar="PATH",
        help="Path to policies.yaml. Activates policy mode: each policy defines "
             "the exact post stance and the opposing stance for hate-speech comments. "
             "Replaces the standard topic×stance design matrix. "
             "Example: configs/policies.yaml",
    )
    parser.add_argument(
        "--from-csv", type=Path, default=None, metavar="PATH",
        help="Path to an R-generated stimulus design CSV. When provided, posts and "
             "comments are generated from this design instead of the internal sampler. "
             "Example: data/stim_df_italy.csv",
    )
    parser.add_argument(
        "--country", type=str, default=None,
        metavar="COUNTRY",
        help="Country of origin for all profiles in this run "
             "(Belgium|Italy|Spain|France|Germany). "
             "Auto-detected from the CSV filename when omitted.",
    )
    return parser.parse_args()


async def _render_from_jsonl(args: argparse.Namespace) -> None:
    """Render post stimuli HTML + profile pages from an existing stimuli.jsonl."""
    import json
    from src.generators.html_stimuli import generate_html_stimuli, take_screenshots
    from src.generators.html_profiles import generate_profile_pages
    from src.models import StimulusRow

    jsonl_path = args.from_jsonl.resolve()
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found.", file=sys.stderr)
        sys.exit(1)

    output_dir = (
        jsonl_path.parent
        if args.output_dir == Path("outputs/run_001")
        else args.output_dir
    )

    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file  = output_dir / "logs" / "render.log"
    setup_logging(level=log_level, log_file=log_file)
    log = logging.getLogger("pipeline.render")

    rows = [
        StimulusRow.model_validate(json.loads(line))
        for line in jsonl_path.open(encoding="utf-8")
        if line.strip()
    ]
    log.info("Loaded %d rows from %s", len(rows), jsonl_path)

    # ── Post stimulus pages (9 engagement variants each) ──────────────────────
    html_paths = generate_html_stimuli(rows, output_dir, _ROOT, args.seed)

    # ── Profile pages (one per unique profile) ────────────────────────────────
    seen: set[str] = set()
    unique_profile_pairs: list[tuple[StimulusRow, str]] = []
    for row in rows:
        if row.profile_id not in seen:
            seen.add(row.profile_id)
            topic = row.topic.value if hasattr(row.topic, "value") else str(row.topic)
            unique_profile_pairs.append((row, topic))
    profile_html_paths = generate_profile_pages(unique_profile_pairs, output_dir, _ROOT, args.seed)

    if args.screenshots:
        png_dir         = output_dir / "png"
        png_profiles_dir = output_dir / "png_profiles"
        await take_screenshots(html_paths, png_dir)
        await take_screenshots(profile_html_paths, png_profiles_dir)
        print(
            f"Done.\n"
            f"  {len(html_paths)} post PNGs  → {png_dir}/\n"
            f"  {len(profile_html_paths)} profile PNGs → {png_profiles_dir}/"
        )
    else:
        print(
            f"Done.\n"
            f"  {len(html_paths)} post HTML files  → {output_dir}/html/\n"
            f"  {len(profile_html_paths)} profile HTML files → {output_dir}/html_profiles/"
        )


async def main() -> None:
    args = parse_args()

    # ── Standalone render mode (no LLM calls) ─────────────────────────────────
    if args.from_jsonl:
        await _render_from_jsonl(args)
        return

    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file  = args.output_dir / "logs" / "pipeline.log"
    setup_logging(level=log_level, log_file=log_file)

    log = logging.getLogger("pipeline.main")

    # Validate API key presence for the chosen backend
    if not args.dry_run:
        is_claude = args.model.startswith("claude")
        if is_claude and not os.getenv("ANTHROPIC_API_KEY"):
            log.error("ANTHROPIC_API_KEY is not set. Add it to .env or export it.")
            sys.exit(1)
        if not is_claude and not os.getenv("OPENAI_API_KEY"):
            log.error("OPENAI_API_KEY is not set. Add it to .env or export it.")
            sys.exit(1)

    study_cfg = load_yaml(args.config)
    policies_cfg = load_yaml(args.policies) if args.policies else None
    if policies_cfg:
        log.info("Policy mode: loaded %d policies from %s",
                 len(policies_cfg.get("policies", [])), args.policies)

    cfg = GenerationConfig(
        study_cfg=study_cfg,
        n_profiles=args.n_profiles,
        seed=args.seed,
        output_dir=args.output_dir,
        model=args.model,
        enable_severity_judge=not args.no_judge,
        enable_realism_check=not args.no_realism,
        reject_on_mismatch=not args.no_reject,
        dry_run=args.dry_run,
        generate_html=args.html or args.screenshots,
        screenshots=args.screenshots,
        policies_cfg=policies_cfg,
        csv_path=getattr(args, "from_csv", None),
        country=getattr(args, "country", None) or "",
    )

    manifest = await run_pipeline(cfg, configs_dir=CONFIGS_DIR)

    if not args.dry_run:
        print(f"\nDone. {manifest.n_passed} stimuli written to {args.output_dir}/final/")


if __name__ == "__main__":
    asyncio.run(main())
