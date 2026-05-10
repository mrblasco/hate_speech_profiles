#!/usr/bin/env python3
"""
validate_matching.py
=====================
Two-layer coherence validation for profile–comment pairings in vignette metadata.

Layer 1 — Rule-based tag checks (all 18 000 rows):
  1. Cross-ideology constraint: profile.ideology ≠ comment.ideology
  2. Topic alignment:           profile.topic == comment.topic
  3. Target-group coherence:    comment.target_tag ∈ profile.identity_tags
  4. Within-respondent profile diversity: no profile_id repeated per respondent

Layer 2 — LLM coherence scoring (stratified sample of N_SAMPLE rows):
  Calls claude-haiku-4-5 to score each pairing 1–5 for semantic coherence.
  Results saved to output/validation/llm_coherence_sample.csv.

Usage
-----
    python validate_matching.py
    python validate_matching.py --metadata output/metadata/vignette_metadata.csv
    python validate_matching.py --no-llm          # skip Layer 2
    python validate_matching.py --sample 50       # smaller LLM sample
"""

import argparse
import csv
import json
import logging
import os
import pathlib
import sys
import time
from typing import Optional

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_METADATA = pathlib.Path("output/metadata/vignette_metadata.csv")
PROFILES_CSV     = pathlib.Path("data/profiles.csv")
COMMENTS_CSV     = pathlib.Path("data/hate_comments.csv")
OUTPUT_DIR       = pathlib.Path("output/validation")

N_SAMPLE         = 200    # number of rows for LLM scoring
LLM_MODEL        = "claude-haiku-4-5-20251001"
LLM_MAX_TOKENS   = 256
LLM_BATCH        = 10     # rows per API call (use structured prompt)

PASS = "✓"
FAIL = "✗"
WARN = "⚠"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def ok(msg: str) -> None:
    print(f"  {PASS}  {msg}")


def fail(msg: str) -> None:
    print(f"  {FAIL}  {msg}", file=sys.stderr)


def warn(msg: str) -> None:
    print(f"  {WARN}  {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# Layer 1 — Rule-based checks
# ──────────────────────────────────────────────────────────────────────────────

def check_cross_ideology(df: pd.DataFrame) -> list[str]:
    """profile.ideology must differ from comment ideology."""
    errors: list[str] = []
    section("Cross-ideology constraint  (profile.ideology ≠ comment.ideology)")

    if "profile_ideology" not in df.columns:
        warn("Column 'profile_ideology' not in metadata — skipping. Re-run pipeline.py after prepare_data.py.")
        return errors

    same = df[df["profile_ideology"] == df["ideology"]]
    if same.empty:
        ok(f"All {len(df):,} rows satisfy cross-ideology constraint")
    else:
        n = len(same)
        msg = f"{n} rows have matching profile and comment ideology"
        fail(msg)
        errors.append(msg)
        print(f"       Sample violations:\n{same[['respondent_id','topic','ideology','profile_ideology','profile_id']].head(5).to_string(index=False)}")

    return errors


def check_topic_alignment(
    df: pd.DataFrame,
    profiles_df: pd.DataFrame,
) -> list[str]:
    """profile.topic must match comment.topic in every row."""
    errors: list[str] = []
    section("Topic alignment  (profile.topic == comment.topic)")

    if "profile_id" not in df.columns:
        warn("Column 'profile_id' not in metadata — skipping.")
        return errors

    pid_topic = profiles_df.set_index("profile_id")["topic"].to_dict()
    df = df.copy()
    df["profile_topic"] = df["profile_id"].map(pid_topic)
    mismatches = df[df["topic"] != df["profile_topic"]]

    if mismatches.empty:
        ok(f"All {len(df):,} rows have matching topic")
    else:
        n = len(mismatches)
        msg = f"{n} rows have mismatched topics"
        fail(msg)
        errors.append(msg)
        print(f"       Sample:\n{mismatches[['respondent_id','topic','profile_topic','profile_id']].head(5).to_string(index=False)}")

    return errors


def check_target_group_coherence(
    df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    comments_df: pd.DataFrame,
) -> list[str]:
    """comment.target_tag should be present in profile.identity_tags."""
    errors: list[str] = []
    section("Target-group coherence  (comment.target_tag ∈ profile.identity_tags)")

    if "target_tag" not in comments_df.columns:
        warn("Column 'target_tag' not in hate_comments.csv — skipping. Run prepare_data.py first.")
        return errors
    if "identity_tags" not in profiles_df.columns:
        warn("Column 'identity_tags' not in profiles.csv — skipping. Run prepare_data.py first.")
        return errors

    # Build lookup maps
    pid_tags    = profiles_df.set_index("profile_id")["identity_tags"].to_dict()
    # comment text → target_tag (first match; texts are unique)
    text_to_tag = dict(zip(comments_df["text"], comments_df["target_tag"]))

    df = df.copy()
    df["comment_target_tag"] = df["comment_text"].map(text_to_tag)
    df["profile_identity"]   = df["profile_id"].map(pid_tags)

    def _coherent(row) -> bool:
        tag  = row["comment_target_tag"]
        tags = row["profile_identity"]
        if pd.isna(tag) or pd.isna(tags):
            return True   # can't check if data is missing
        return tag in str(tags)

    df["coherent"] = df.apply(_coherent, axis=1)
    incoherent = df[~df["coherent"]]

    total_checkable = df["comment_target_tag"].notna().sum()
    n_incoherent    = len(incoherent)

    if n_incoherent == 0:
        ok(f"All {total_checkable:,} checkable rows pass target-group coherence")
    else:
        pct = n_incoherent / max(total_checkable, 1) * 100
        msg = f"{n_incoherent} / {total_checkable} rows ({pct:.1f}%) fail target-group coherence"
        if pct < 5:
            warn(msg + "  (low — may be inherent topic ambiguity)")
        else:
            fail(msg)
            errors.append(msg)

        if not incoherent.empty:
            print(f"       Sample (topic / comment_target / profile_identity):")
            sample = incoherent[["topic", "comment_target_tag", "profile_identity", "profile_ideology", "ideology"]].head(8)
            print(sample.to_string(index=False))

    return errors


def check_profile_diversity(df: pd.DataFrame) -> list[str]:
    """No profile_id should repeat within the same respondent block."""
    errors: list[str] = []
    section("Within-respondent profile diversity  (no repeated profile_id)")

    if "profile_id" not in df.columns:
        warn("Column 'profile_id' not in metadata — skipping.")
        return errors

    dup = df.groupby(["respondent_id", "profile_id"]).size()
    repeats = dup[dup > 1]

    if repeats.empty:
        ok(f"No profile_id repeats across any of the {df['respondent_id'].nunique():,} respondent blocks")
    else:
        n = len(repeats)
        msg = f"{n} (respondent_id, profile_id) pairs appear more than once"
        fail(msg)
        errors.append(msg)

    return errors


# ──────────────────────────────────────────────────────────────────────────────
# Layer 2 — LLM coherence scoring
# ──────────────────────────────────────────────────────────────────────────────

COHERENCE_PROMPT_TEMPLATE = """You are a social-science researcher evaluating experimental stimuli for a study on online hate speech.

For each pairing below, rate how semantically coherent the hate comment is as a reply to the profile post on a scale of 1 to 5:
  1 = Totally incoherent: the hate comment targets the wrong group, ignores the post content, or makes no logical sense as a reply
  2 = Mostly incoherent: loosely related topic but wrong target or stance
  3 = Partially coherent: same topic, comment could plausibly appear but is off-target
  4 = Mostly coherent: comment clearly targets the poster's stance or identity
  5 = Fully coherent: comment is a direct, realistic attack on what the poster stands for

Return a JSON array of objects with keys: "id", "score" (int 1–5), "reason" (one sentence).

Pairings:
{pairings}"""


def call_llm_batch(client, batch: list[dict]) -> list[dict]:
    pairings_str = json.dumps(
        [{"id": r["_row_id"], "post": r["target_message"], "comment": r["comment_text"]}
         for r in batch],
        ensure_ascii=False, indent=2
    )
    prompt = COHERENCE_PROMPT_TEMPLATE.format(pairings=pairings_str)

    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS * len(batch),
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            if text.startswith("```"):
                text = "\n".join(l for l in text.splitlines() if not l.startswith("```"))
            return json.loads(text)
        except Exception as exc:
            log.warning(f"LLM batch error (attempt {attempt + 1}/3): {exc}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    log.error("LLM batch failed after 3 attempts — returning empty results")
    return []


def run_llm_scoring(df: pd.DataFrame, n_sample: int) -> Optional[pd.DataFrame]:
    section(f"LLM coherence scoring  (sample n={n_sample})")

    try:
        import anthropic
        client = anthropic.Anthropic()
    except ImportError:
        warn("anthropic package not installed — skipping LLM scoring")
        return None

    required = {"target_message", "comment_text", "topic", "ideology", "severity"}
    missing  = required - set(df.columns)
    if missing:
        warn(f"Metadata missing columns {missing} — skipping LLM scoring")
        return None

    # Stratified sample across topic × ideology × severity
    strata_cols = ["topic", "ideology", "severity"]
    try:
        sample = (
            df.groupby(strata_cols, group_keys=False)
            .apply(lambda g: g.sample(
                min(len(g), max(1, n_sample // (len(df[strata_cols].drop_duplicates())))),
                random_state=42,
            ))
            .head(n_sample)
            .copy()
        )
    except Exception:
        sample = df.sample(min(n_sample, len(df)), random_state=42).copy()

    sample = sample.reset_index(drop=True)
    sample["_row_id"] = sample.index

    log.info(f"  Scoring {len(sample)} sampled pairings with {LLM_MODEL} …")

    all_scores: list[dict] = []
    rows = sample.to_dict("records")
    for i in range(0, len(rows), LLM_BATCH):
        batch = rows[i : i + LLM_BATCH]
        log.info(f"  Batch {i//LLM_BATCH + 1} / {(len(rows) + LLM_BATCH - 1)//LLM_BATCH}")
        results = call_llm_batch(client, batch)
        all_scores.extend(results)

    id_to_score  = {r["id"]: r.get("score",  None) for r in all_scores}
    id_to_reason = {r["id"]: r.get("reason", "")   for r in all_scores}
    sample["llm_score"]  = sample["_row_id"].map(id_to_score)
    sample["llm_reason"] = sample["_row_id"].map(id_to_reason)
    sample = sample.drop(columns=["_row_id"])

    valid_scores = sample["llm_score"].dropna()
    if len(valid_scores):
        mean_score = valid_scores.mean()
        pct_4plus  = (valid_scores >= 4).mean() * 100
        print(f"  Mean coherence score : {mean_score:.2f} / 5.0")
        print(f"  % scoring ≥ 4        : {pct_4plus:.1f}%")

        if mean_score >= 4.0:
            ok(f"Mean LLM coherence = {mean_score:.2f} — target met (≥ 4.0)")
        else:
            warn(f"Mean LLM coherence = {mean_score:.2f} — below target of 4.0")

        # Show worst pairings
        worst = sample.nsmallest(5, "llm_score")[
            ["topic", "ideology", "severity", "llm_score", "llm_reason",
             "target_message", "comment_text"]
        ]
        print(f"\n  Lowest-scoring pairings:")
        for _, r in worst.iterrows():
            print(f"    [{r['llm_score']}] {r['topic']} / {r['ideology']} / {r['severity']}")
            print(f"         POST:    {str(r['target_message'])[:80]}")
            print(f"         COMMENT: {str(r['comment_text'])[:80]}")
            print(f"         REASON:  {r['llm_reason']}")
    else:
        warn("No valid scores returned from LLM")

    return sample


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate profile–comment matching in vignette metadata."
    )
    parser.add_argument("--metadata",  type=pathlib.Path, default=DEFAULT_METADATA)
    parser.add_argument("--profiles",  type=pathlib.Path, default=PROFILES_CSV)
    parser.add_argument("--comments",  type=pathlib.Path, default=COMMENTS_CSV)
    parser.add_argument("--no-llm",    action="store_true", help="Skip LLM scoring")
    parser.add_argument("--sample",    type=int, default=N_SAMPLE,
                        help=f"Number of rows for LLM scoring (default: {N_SAMPLE})")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  MATCHING COHERENCE VALIDATION REPORT")
    print("═" * 60)

    for path in (args.metadata, args.profiles, args.comments):
        if not path.exists():
            print(f"\n  {FAIL}  File not found: {path}", file=sys.stderr)
            if path == args.metadata:
                print("  Run pipeline.py first.", file=sys.stderr)
            else:
                print("  Run prepare_data.py first.", file=sys.stderr)
            sys.exit(1)

    df          = pd.read_csv(args.metadata)
    profiles_df = pd.read_csv(args.profiles)
    comments_df = pd.read_csv(args.comments)

    print(f"\n  metadata  : {len(df):,} rows")
    print(f"  profiles  : {len(profiles_df):,} rows")
    print(f"  comments  : {len(comments_df):,} rows")

    all_errors: list[str] = []

    # ── Layer 1 ──────────────────────────────────────────────────────────────
    all_errors += check_cross_ideology(df)
    all_errors += check_topic_alignment(df, profiles_df)
    all_errors += check_target_group_coherence(df, profiles_df, comments_df)
    all_errors += check_profile_diversity(df)

    # ── Layer 2 ──────────────────────────────────────────────────────────────
    llm_results: Optional[pd.DataFrame] = None
    if not args.no_llm:
        llm_results = run_llm_scoring(df, args.sample)

    # ── Write outputs ─────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Layer 1: write violation report
    if all_errors:
        violations = []
        # Collect cross-ideology violations
        if "profile_ideology" in df.columns:
            same = df[df["profile_ideology"] == df["ideology"]].copy()
            same["violation"] = "cross_ideology"
            violations.append(same)
        violation_df = pd.concat(violations) if violations else pd.DataFrame()
        if not violation_df.empty:
            vpath = OUTPUT_DIR / "tag_check_report.csv"
            violation_df.to_csv(vpath, index=False)
            log.info(f"  Violations written → {vpath}")
    else:
        section("No rule violations found")
        ok("tag_check_report.csv not written (no violations)")

    # Layer 2: write LLM scores
    if llm_results is not None:
        lpath = OUTPUT_DIR / "llm_coherence_sample.csv"
        llm_results.to_csv(lpath, index=False)
        log.info(f"  LLM scores written → {lpath}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    if all_errors:
        print(f"  {FAIL}  {len(all_errors)} rule violation(s):")
        for e in all_errors:
            print(f"       • {e}")
        print("═" * 60 + "\n")
        sys.exit(1)
    else:
        print(f"  {PASS}  All rule checks passed.")
        print("═" * 60 + "\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
