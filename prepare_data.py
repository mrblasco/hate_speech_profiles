#!/usr/bin/env python3
"""
prepare_data.py
===============
Annotates existing input CSVs with columns required for cross-ideological
matching and semantic coherence validation.

Adds to data/profiles.csv:
  - ideology:       'progressive' or 'conservative'
  - identity_tags:  comma-separated group membership tags

Adds to data/hate_comments.csv:
  - target_tag:  the group/identity the comment attacks

Run once after cloning / before generate_conservative_profiles.py.
Idempotent — skips columns that already exist.

Usage
-----
    python prepare_data.py
    python prepare_data.py --data-dir data/countries/it   # for country variants
"""

import argparse
import csv
import pathlib
import sys

# ──────────────────────────────────────────────────────────────────────────────
# Profile annotations
# ──────────────────────────────────────────────────────────────────────────────

# profile_id → (ideology, identity_tags)
# Ideology: 'progressive' = minority/progressive-aligned poster
#           'conservative' = majority/conservative-aligned poster
# Identity tags: what group(s) the poster represents (for coherence checking)
PROFILE_METADATA: dict[str, tuple[str, str]] = {
    # ── Immigration / migrants ── all progressive (minority immigrants) ────────
    "P001": ("progressive", "immigrant"),
    "P002": ("progressive", "immigrant"),
    "P003": ("progressive", "immigrant"),
    "P004": ("progressive", "immigrant"),
    "P005": ("progressive", "immigrant"),
    "P006": ("progressive", "immigrant"),
    # ── Sexual orientation / gender identity ── all progressive (LGBTQ) ───────
    "P007": ("progressive", "LGBTQ"),
    "P008": ("progressive", "LGBTQ"),
    "P009": ("progressive", "LGBTQ"),
    "P010": ("progressive", "LGBTQ"),
    "P011": ("progressive", "LGBTQ"),
    "P012": ("progressive", "LGBTQ"),
    # ── Religion (Muslim / Jewish) ── progressive minority believers ──────────
    "P013": ("progressive", "Muslim"),
    "P014": ("progressive", "Muslim"),
    "P015": ("progressive", "Muslim"),
    "P016": ("progressive", "Muslim"),
    "P017": ("progressive", "Jewish"),
    "P018": ("progressive", "Muslim"),
    # ── Gender issues (misogyny) ── all progressive (feminist women) ──────────
    "P019": ("progressive", "feminist_woman"),
    "P020": ("progressive", "feminist_woman"),
    "P021": ("progressive", "feminist_woman"),
    "P022": ("progressive", "feminist_woman"),
    "P023": ("progressive", "feminist_woman"),
    "P024": ("progressive", "feminist_woman"),
    # ── Racism / ethnicity ── progressive (African-American) ──────────────────
    "P025": ("progressive", "ethnic_minority"),
    "P026": ("progressive", "ethnic_minority"),
    "P027": ("progressive", "ethnic_minority"),
    "P028": ("progressive", "ethnic_minority"),
    "P029": ("progressive", "ethnic_minority"),
    "P030": ("progressive", "ethnic_minority"),
    # ── Nationalism / identity politics ── MIXED ──────────────────────────────
    "P031": ("conservative", "nationalist"),    # "National sovereignty...celebrate"
    "P032": ("progressive",  "globalist"),      # "global citizen"
    "P033": ("progressive",  "globalist"),      # "cosmopolitan"
    "P034": ("conservative", "nationalist"),    # "Proud American"
    "P035": ("progressive",  "globalist"),      # "global citizen"
    "P036": ("progressive",  "globalist"),      # "model UN"
    # ── Immigration / migrants (EU set) ── progressive ────────────────────────
    "P037": ("progressive", "immigrant"),
    "P038": ("progressive", "immigrant"),
    "P039": ("progressive", "immigrant"),
    "P040": ("progressive", "immigrant"),
    "P041": ("progressive", "immigrant"),
    "P042": ("progressive", "immigrant"),
    # ── Sexual orientation (EU set) ── progressive ────────────────────────────
    "P043": ("progressive", "LGBTQ"),
    "P044": ("progressive", "LGBTQ"),
    "P045": ("progressive", "LGBTQ"),
    "P046": ("progressive", "LGBTQ"),
    "P047": ("progressive", "LGBTQ"),
    "P048": ("progressive", "LGBTQ"),
    # ── Religion (EU set) ── progressive ─────────────────────────────────────
    "P049": ("progressive", "Muslim"),
    "P050": ("progressive", "Muslim"),
    "P051": ("progressive", "Muslim"),
    "P052": ("progressive", "Muslim"),
    "P053": ("progressive", "Jewish"),
    "P054": ("progressive", "Muslim"),
    # ── Gender (EU set) ── progressive ───────────────────────────────────────
    "P055": ("progressive", "feminist_woman"),
    "P056": ("progressive", "feminist_woman"),
    "P057": ("progressive", "feminist_woman"),
    "P058": ("progressive", "feminist_woman"),
    "P059": ("progressive", "feminist_woman"),
    "P060": ("progressive", "feminist_woman"),
    # ── Racism / ethnicity (EU set) ── progressive (Afro-European) ───────────
    "P061": ("progressive", "ethnic_minority"),
    "P062": ("progressive", "ethnic_minority"),
    "P063": ("progressive", "ethnic_minority"),
    "P064": ("progressive", "ethnic_minority"),
    "P065": ("progressive", "ethnic_minority"),
    "P066": ("progressive", "ethnic_minority"),
    # ── Nationalism (EU set) ── MIXED ─────────────────────────────────────────
    "P067": ("conservative", "nationalist"),    # "España primero"
    "P068": ("progressive",  "globalist"),      # "europeista"
    "P069": ("conservative", "nationalist"),    # "Heimat matters"
    "P070": ("progressive",  "globalist"),      # "EU generation"
    "P071": ("conservative", "nationalist"),    # "Italia prima"
    "P072": ("progressive",  "globalist"),      # "Weltbürgerin"
}

# ──────────────────────────────────────────────────────────────────────────────
# Hate-comment target tags
# ──────────────────────────────────────────────────────────────────────────────
# Ordered to match rows in hate_comments.csv (18 rows per topic, in the order:
#   con-opinion×3, prog-opinion×3, con-dehumanising×3, prog-dehumanising×3,
#   con-incitement×3, prog-incitement×3).
#
# Tags identify WHICH group the comment targets, so the validator can verify
# that the matched profile's identity_tags contains the attack target.
#
# Conservative tags  → minority/marginalised groups (the poster's own group)
# Progressive tags   → majority/dominant groups (the poster's own group when
#                      conservative profiles are added)

COMMENT_TARGET_TAGS: list[str] = [
    # ── Racism / ethnicity ────────────────────────────────────────────────────
    "ethnic_minority", "ethnic_minority", "ethnic_minority",           # con-op
    "white_European_nationalist", "white_European_nationalist", "white_European_nationalist",  # prog-op
    "ethnic_minority", "ethnic_minority", "ethnic_minority",           # con-deh
    "white_European_nationalist", "white_European_nationalist", "white_European_nationalist",  # prog-deh
    "ethnic_minority", "ethnic_minority", "ethnic_minority",           # con-inc
    "white_European_nationalist", "white_European_nationalist", "white_European_nationalist",  # prog-inc
    # ── Religion (Muslim / Jewish) ────────────────────────────────────────────
    # con-op: "Islam incompatible", "Jewish organisations lobby", "Muslim immigration"
    "Muslim", "Jewish", "Muslim",                                      # con-op
    "religious_conservative", "religious_conservative", "religious_conservative",  # prog-op
    # con-deh: "Islam death cult", "Jewish elites control", "Muslims savages"
    "Muslim", "Jewish", "Muslim",                                      # con-deh
    "religious_conservative", "religious_conservative", "religious_conservative",  # prog-deh
    # con-inc: all target Islam/mosques
    "Muslim", "Muslim", "Muslim",                                      # con-inc
    "religious_conservative", "religious_conservative", "religious_conservative",  # prog-inc
    # ── Immigration / migrants ────────────────────────────────────────────────
    "immigrant", "immigrant", "immigrant",                             # con-op
    "nativist", "nativist", "nativist",                                # prog-op
    "immigrant", "immigrant", "immigrant",                             # con-deh
    "nativist", "nativist", "nativist",                                # prog-deh
    "immigrant", "immigrant", "immigrant",                             # con-inc
    "nativist", "nativist", "nativist",                                # prog-inc
    # ── Gender issues (misogyny) ──────────────────────────────────────────────
    "feminist_woman", "feminist_woman", "feminist_woman",              # con-op
    "traditional_man", "traditional_man", "traditional_man",          # prog-op
    "feminist_woman", "feminist_woman", "feminist_woman",              # con-deh
    "traditional_man", "traditional_man", "traditional_man",          # prog-deh
    "feminist_woman", "feminist_woman", "feminist_woman",              # con-inc
    "traditional_man", "traditional_man", "traditional_man",          # prog-inc
    # ── Sexual orientation / gender identity ──────────────────────────────────
    "LGBTQ", "LGBTQ", "LGBTQ",                                        # con-op
    "anti_LGBTQ", "anti_LGBTQ", "anti_LGBTQ",                        # prog-op
    "LGBTQ", "LGBTQ", "LGBTQ",                                        # con-deh
    "anti_LGBTQ", "anti_LGBTQ", "anti_LGBTQ",                        # prog-deh
    "LGBTQ", "LGBTQ", "LGBTQ",                                        # con-inc
    "anti_LGBTQ", "anti_LGBTQ", "anti_LGBTQ",                        # prog-inc
    # ── Nationalism / identity politics ───────────────────────────────────────
    # con-op: targets globalists ("national sovereignty", "culture worth protecting", "EU eroding democracy")
    "globalist", "globalist", "globalist",                             # con-op
    "nationalist", "nationalist", "nationalist",                       # prog-op
    # con-deh: targets foreigners explicitly ("not our equals", "no real roots", "contaminating presence")
    "ethnic_minority", "ethnic_minority", "ethnic_minority",           # con-deh
    "nationalist", "nationalist", "nationalist",                       # prog-deh
    # con-inc: targets foreigners ("refuse to integrate...removed")
    "ethnic_minority", "ethnic_minority", "ethnic_minority",           # con-inc
    "nationalist", "nationalist", "nationalist",                       # prog-inc
]

assert len(COMMENT_TARGET_TAGS) == 108, f"Expected 108 tags, got {len(COMMENT_TARGET_TAGS)}"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def read_csv(path: pathlib.Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return list(fieldnames), rows


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def annotate_profiles(data_dir: pathlib.Path) -> None:
    path = data_dir / "profiles.csv"
    fieldnames, rows = read_csv(path)

    needs_ideology = "ideology" not in fieldnames
    needs_tags     = "identity_tags" not in fieldnames

    if not needs_ideology and not needs_tags:
        print(f"  profiles.csv: already annotated — skipping")
        return

    unknown = []
    for row in rows:
        pid = row["profile_id"]
        if pid not in PROFILE_METADATA:
            unknown.append(pid)
            continue
        ideology, tags = PROFILE_METADATA[pid]
        if needs_ideology:
            row["ideology"] = ideology
        if needs_tags:
            row["identity_tags"] = tags

    if unknown:
        print(f"  WARNING: {len(unknown)} unknown profile_ids — no annotation added: {unknown[:5]}")

    if needs_ideology and "ideology" not in fieldnames:
        fieldnames.append("ideology")
    if needs_tags and "identity_tags" not in fieldnames:
        fieldnames.append("identity_tags")

    write_csv(path, fieldnames, rows)
    added = []
    if needs_ideology: added.append("ideology")
    if needs_tags:     added.append("identity_tags")
    print(f"  profiles.csv: added columns {added}")


def annotate_comments(data_dir: pathlib.Path) -> None:
    path = data_dir / "hate_comments.csv"
    fieldnames, rows = read_csv(path)

    if "target_tag" in fieldnames:
        print(f"  hate_comments.csv: already annotated — skipping")
        return

    if len(rows) != 108:
        print(f"  WARNING: expected 108 comment rows, got {len(rows)} — skipping annotation")
        return

    for row, tag in zip(rows, COMMENT_TARGET_TAGS):
        row["target_tag"] = tag

    fieldnames.append("target_tag")
    write_csv(path, fieldnames, rows)
    print(f"  hate_comments.csv: added column 'target_tag'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate input CSVs with ideology and target-group tags."
    )
    parser.add_argument(
        "--data-dir", type=pathlib.Path, default=pathlib.Path("data"),
        help="Directory containing profiles.csv and hate_comments.csv (default: data/)"
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"ERROR: {args.data_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"\nAnnotating data in {args.data_dir}/")
    annotate_profiles(args.data_dir)
    annotate_comments(args.data_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
