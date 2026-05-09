#!/usr/bin/env python3
"""
Factorial Balance Verification
================================
Loads vignette_metadata.csv and checks distributional balance across all
experimental factors.  Prints a summary report and exits with code 1 if
any critical constraint is violated.

Usage
-----
    python verify_balance.py
    python verify_balance.py --metadata output/metadata/vignette_metadata.csv
    python verify_balance.py --tolerance 0.05   # allow ±5 % deviation
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_METADATA = Path("output/metadata/vignette_metadata.csv")

N_RESPONDENTS             = 3_000
N_VIGNETTES_PER_RESP      = 6
EXPECTED_TOTAL            = N_RESPONDENTS * N_VIGNETTES_PER_RESP   # 18 000

TOPICS           = ["Racism / ethnicity", "Religion (Muslim / Jewish)", "Immigration / migrants",
                    "Gender issues (misogyny)", "Sexual orientation / gender identity",
                    "Nationalism / identity politics"]
SEVERITIES       = ["opinion", "dehumanising", "incitement"]
IDEOLOGIES       = ["conservative", "progressive"]
AGE_GROUPS       = ["young_adult", "middle_adult"]
ENGAGEMENT_LEVELS = ["low", "medium", "high"]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

PASS = "✓"
FAIL = "✗"


def _bar(value: float, width: int = 20) -> str:
    """Simple text progress bar for proportions."""
    filled = round(value * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {value:.1%}"


def check_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def check_pass(label: str, detail: str = "") -> None:
    print(f"  {PASS}  {label}" + (f"  ({detail})" if detail else ""))


def check_fail(label: str, detail: str = "") -> None:
    print(f"  {FAIL}  {label}" + (f"  — {detail}" if detail else ""), file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Verification functions
# ──────────────────────────────────────────────────────────────────────────────

def verify_row_count(df: pd.DataFrame) -> list[str]:
    """Check total row count matches expectation."""
    errors = []
    check_section("Row count")
    n = len(df)
    if n == EXPECTED_TOTAL:
        check_pass(f"Total rows: {n:,}")
    else:
        msg = f"Expected {EXPECTED_TOTAL:,}, got {n:,}"
        check_fail("Total rows", msg)
        errors.append(msg)
    return errors


def verify_respondent_structure(df: pd.DataFrame) -> list[str]:
    """
    Every respondent must have exactly N_VIGNETTES_PER_RESP rows,
    each with a unique topic and a unique vignette_order (1–6).
    """
    errors = []
    check_section("Respondent structure")

    # ── Vignette count per respondent ────────────────────────────────────────
    counts = df.groupby("respondent_id").size()
    bad    = counts[counts != N_VIGNETTES_PER_RESP]
    if bad.empty:
        check_pass(f"All {N_RESPONDENTS:,} respondents have exactly {N_VIGNETTES_PER_RESP} vignettes")
    else:
        msg = f"{len(bad)} respondents do not have exactly {N_VIGNETTES_PER_RESP} vignettes"
        check_fail("Vignette count", msg)
        errors.append(msg)

    # ── No repeated topic within a respondent ────────────────────────────────
    topic_counts = df.groupby(["respondent_id", "topic"]).size()
    dup_topics   = topic_counts[topic_counts > 1]
    if dup_topics.empty:
        check_pass("No topic repeats within any respondent block")
    else:
        msg = f"{len(dup_topics)} (respondent, topic) pairs appear more than once"
        check_fail("Topic uniqueness within block", msg)
        errors.append(msg)

    # ── Vignette order values ─────────────────────────────────────────────────
    order_ok = (
        df.groupby("respondent_id")["vignette_order"]
        .apply(lambda s: set(s) == set(range(1, N_VIGNETTES_PER_RESP + 1)))
        .all()
    )
    if order_ok:
        check_pass("Vignette order 1–6 present for every respondent")
    else:
        msg = "Some respondents have incorrect vignette_order values"
        check_fail("Vignette order", msg)
        errors.append(msg)

    return errors


def verify_marginal_balance(
    df: pd.DataFrame,
    tolerance: float,
) -> list[str]:
    """
    Check that each factor level appears at the expected rate (±tolerance).

    Expected rates
    --------------
    topic          : 1/6  ≈ 16.7 %  (each topic appears N_RESPONDENTS times)
    severity       : 1/3  ≈ 33.3 %
    ideology       : 1/2  = 50.0 %
    age_group      : 1/2  = 50.0 %
    engagement_level : 1/3 ≈ 33.3 %
    """
    errors = []
    check_section(f"Marginal balance  (tolerance ±{tolerance:.0%})")

    checks = [
        ("topic",           TOPICS,            1 / len(TOPICS)),
        ("severity",        SEVERITIES,        1 / len(SEVERITIES)),
        ("ideology",        IDEOLOGIES,        1 / len(IDEOLOGIES)),
        ("age_group",       AGE_GROUPS,        1 / len(AGE_GROUPS)),
        ("engagement_level", ENGAGEMENT_LEVELS, 1 / len(ENGAGEMENT_LEVELS)),
    ]

    for col, levels, expected_p in checks:
        counts = df[col].value_counts(normalize=True)
        all_ok = True
        for level in levels:
            actual = counts.get(level, 0.0)
            deviation = abs(actual - expected_p)
            bar = _bar(actual)
            if deviation <= tolerance:
                print(f"     {col}={level:<20} {bar}")
            else:
                msg = (f"{col}={level}: expected {expected_p:.1%}, "
                       f"got {actual:.1%} (Δ={deviation:.1%} > {tolerance:.0%})")
                print(f"     {FAIL} {col}={level:<20} {bar}  ← OUT OF RANGE")
                errors.append(msg)
                all_ok = False
        if all_ok:
            check_pass(f"{col} — all levels within ±{tolerance:.0%}")
        print()

    return errors


def verify_joint_cells(df: pd.DataFrame, tolerance: float) -> list[str]:
    """
    Check full factorial cell coverage:
    topic × severity × ideology × age_group × engagement_level = 216 cells.
    Each cell should appear at least once and counts should be roughly equal.
    """
    errors = []
    check_section("Joint cell coverage  (216 cells)")

    cell_counts = df.groupby(
        ["topic", "severity", "ideology", "age_group", "engagement_level"]
    ).size()

    n_cells_total   = len(TOPICS) * len(SEVERITIES) * len(IDEOLOGIES) * len(AGE_GROUPS) * len(ENGAGEMENT_LEVELS)
    n_cells_found   = len(cell_counts)
    n_cells_missing = n_cells_total - n_cells_found

    if n_cells_missing == 0:
        check_pass(f"All {n_cells_total} cells observed")
    else:
        msg = f"{n_cells_missing}/{n_cells_total} cells have zero observations"
        check_fail("Cell coverage", msg)
        errors.append(msg)

    expected_per_cell = EXPECTED_TOTAL / n_cells_total
    cv = cell_counts.std() / cell_counts.mean()

    print(f"  Expected rows per cell : {expected_per_cell:.1f}")
    print(f"  Observed min / mean / max : "
          f"{cell_counts.min()} / {cell_counts.mean():.1f} / {cell_counts.max()}")
    print(f"  Coefficient of variation  : {cv:.3f}  "
          f"(0 = perfectly balanced)")

    if cv < 0.10:
        check_pass(f"Cell CV={cv:.3f} — well balanced")
    elif cv < 0.20:
        check_pass(f"Cell CV={cv:.3f} — acceptable balance")
    else:
        msg = f"Cell CV={cv:.3f} — high imbalance"
        check_fail("Cell balance", msg)
        errors.append(msg)

    return errors


def verify_stimulus_coverage(df: pd.DataFrame) -> list[str]:
    """Check stimulus filename is non-null for every row."""
    errors = []
    check_section("Stimulus filename coverage")

    n_missing = df["stimulus_filename"].isna().sum()
    if n_missing == 0:
        check_pass("All rows have a stimulus_filename")
    else:
        msg = f"{n_missing} rows missing stimulus_filename"
        check_fail("Stimulus filename", msg)
        errors.append(msg)

    n_unique = df["stimulus_filename"].nunique()
    check_pass(f"Unique stimuli: {n_unique:,}  "
               f"(avg {len(df)/n_unique:.1f} respondents per stimulus)")
    return errors


def print_summary_table(df: pd.DataFrame) -> None:
    """Print a concise cross-tabulation summary."""
    check_section("Cross-tab: topic × severity  (row counts)")
    ct = pd.crosstab(df["topic"], df["severity"])[SEVERITIES]
    print(ct.to_string())

    check_section("Cross-tab: topic × ideology  (row counts)")
    ct2 = pd.crosstab(df["topic"], df["ideology"])[IDEOLOGIES]
    print(ct2.to_string())


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify factorial balance of vignette metadata.")
    parser.add_argument(
        "--metadata", type=Path, default=DEFAULT_METADATA,
        help=f"Path to vignette_metadata.csv  (default: {DEFAULT_METADATA})"
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.02,
        help="Max allowed deviation from expected proportion (default: 0.02 = ±2%%)"
    )
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  VIGNETTE BALANCE VERIFICATION REPORT")
    print("═" * 60)

    if not args.metadata.exists():
        print(f"\n  {FAIL}  File not found: {args.metadata}", file=sys.stderr)
        print("  Run pipeline.py first to generate the metadata.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.metadata)
    print(f"\n  Loaded: {args.metadata}  ({len(df):,} rows)")

    all_errors: list[str] = []
    all_errors += verify_row_count(df)
    all_errors += verify_respondent_structure(df)
    all_errors += verify_marginal_balance(df, args.tolerance)
    all_errors += verify_joint_cells(df, args.tolerance)
    all_errors += verify_stimulus_coverage(df)
    print_summary_table(df)

    print("\n" + "═" * 60)
    if all_errors:
        print(f"  {FAIL}  {len(all_errors)} issue(s) found:")
        for e in all_errors:
            print(f"       • {e}")
        print("═" * 60 + "\n")
        sys.exit(1)
    else:
        print(f"  {PASS}  All checks passed — design is balanced.")
        print("═" * 60 + "\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
