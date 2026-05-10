#!/usr/bin/env python3
"""
Instagram-Style Vignette Stimulus Generator
============================================
Generates balanced experimental stimuli (HTML + PNG screenshots) for a
vignette survey on online hate speech.

Factorial design
----------------
    6 topics × 3 severity × 2 ideology × 2 age_groups × 3 engagement = 216 cells
    3 000 respondents × 6 vignettes each = 18 000 rows

Balance strategy
----------------
For every topic we build a pool of N_RESPONDENTS rows that tile all
combinations of (severity × ideology × age_group × engagement_level) as
evenly as possible, then shuffle.  Respondent i draws row i from every
topic's pool, guaranteeing:
  • Each respondent sees each topic exactly once (no topic repeats per block)
  • Marginal distributions are balanced across the full sample

Usage
-----
    python pipeline.py                        # full run
    python pipeline.py --limit 20            # render only 20 screenshots
    python pipeline.py --skip-screenshots    # metadata + HTML only
    python pipeline.py --seed 99             # different seed
"""

import argparse
import asyncio
import itertools
import logging
import math
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42
N_RESPONDENTS = 3_000
N_VIGNETTES_PER_RESPONDENT = 6   # one per topic

# Batch size for concurrent Playwright page renders
SCREENSHOT_BATCH_SIZE = 20

# Country code — overridden at runtime by --country flag
COUNTRY = "en"

# Input / output paths (recalculated in main() when COUNTRY != "en")
DATA_DIR      = Path("data")
OUTPUT_DIR    = Path("output")
HTML_DIR      = OUTPUT_DIR / "html"
PNG_DIR       = OUTPUT_DIR / "png"
METADATA_DIR  = OUTPUT_DIR / "metadata"
TEMPLATES_DIR = Path("templates")
STATIC_DIR    = Path("static")

# ── Factor levels (must match exact values in the CSV files) ─────────────────
TOPICS           = ["Racism / ethnicity", "Religion (Muslim / Jewish)", "Immigration / migrants",
                    "Gender issues (misogyny)", "Sexual orientation / gender identity",
                    "Nationalism / identity politics"]
SEVERITIES       = ["opinion", "dehumanising", "incitement"]
IDEOLOGIES       = ["conservative", "progressive"]
AGE_GROUPS       = ["adolescent", "young_adult"]
ENGAGEMENT_LEVELS = ["low", "medium", "high"]

# Pool of anonymous-looking commenter usernames for the hate comment display
COMMENTER_USERNAMES = [
    "user48291",      "real_talk_99",    "truth_seeker_x",  "anonymous_voice",
    "just_sayin_2024","no_filter_guy",   "freedom_speaker", "real_deal_22",
    "speaking_facts_","open_minded_99",  "debate_this_now", "commentator45",
    "plaintruth2024", "honest_guy_12",   "voice_of_reason", "citizen_free",
    "noreservations_","outsider_view",   "justmyopinion99", "the_watchman_x",
    "critical_thinker","questioning_all","skeptic_online",  "frank_comments",
    "telling_it_real","common_sense_01", "facts_not_feels", "reality_check_",
]

# Gradient backgrounds used in the post image card (one per topic)
TOPIC_GRADIENTS = {
    "Racism / ethnicity":                  "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
    "Religion (Muslim / Jewish)":          "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
    "Immigration / migrants":              "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "Gender issues (misogyny)":            "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
    "Sexual orientation / gender identity":"linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
    "Nationalism / identity politics":     "linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)",
}

# Representative emoji shown on the post-image card
TOPIC_EMOJIS = {
    "Racism / ethnicity":                  "✊🏾",
    "Religion (Muslim / Jewish)":          "🕌",
    "Immigration / migrants":              "🌍",
    "Gender issues (misogyny)":            "✊",
    "Sexual orientation / gender identity":"🏳️‍🌈",
    "Nationalism / identity politics":     "🏴",
}

# Post-age strings to keep the timestamp realistic
# (overridden per country in main())
POST_AGES = ["2 HOURS AGO", "3 HOURS AGO", "4 HOURS AGO",
             "5 HOURS AGO", "6 HOURS AGO", "8 HOURS AGO"]

# Localised UI strings keyed by country code
UI_STRINGS: dict[str, dict] = {
    "en": {
        "likes":    "likes",
        "view_all": "View all {n} comments",
        "view":     "View {n} comments",
        "more":     "… more",
        "post_ages": ["2 HOURS AGO", "3 HOURS AGO", "4 HOURS AGO",
                      "5 HOURS AGO", "6 HOURS AGO", "8 HOURS AGO"],
    },
    "it": {
        "likes":    "Mi piace",
        "view_all": "Visualizza tutti i {n} commenti",
        "view":     "Visualizza {n} commenti",
        "more":     "… altro",
        "post_ages": ["2 ORE FA", "3 ORE FA", "4 ORE FA",
                      "5 ORE FA", "6 ORE FA", "8 ORE FA"],
    },
    "es": {
        "likes":    "Me gusta",
        "view_all": "Ver los {n} comentarios",
        "view":     "Ver {n} comentarios",
        "more":     "… más",
        "post_ages": ["HACE 2 HORAS", "HACE 3 HORAS", "HACE 4 HORAS",
                      "HACE 5 HORAS", "HACE 6 HORAS", "HACE 8 HORAS"],
    },
    "fr": {
        "likes":    "J'aime",
        "view_all": "Voir les {n} commentaires",
        "view":     "Voir {n} commentaires",
        "more":     "… plus",
        "post_ages": ["IL Y A 2 H", "IL Y A 3 H", "IL Y A 4 H",
                      "IL Y A 5 H", "IL Y A 6 H", "IL Y A 8 H"],
    },
    "de": {
        "likes":    "Gefällt mir",
        "view_all": "Alle {n} Kommentare ansehen",
        "view":     "{n} Kommentare ansehen",
        "more":     "… mehr",
        "post_ages": ["VOR 2 STUNDEN", "VOR 3 STUNDEN", "VOR 4 STUNDEN",
                      "VOR 5 STUNDEN", "VOR 6 STUNDEN", "VOR 8 STUNDEN"],
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read and validate the three input CSVs.

    Returns
    -------
    profiles_df, comments_df, engagement_df
    """
    log.info("Loading input data …")

    profiles   = pd.read_csv(DATA_DIR / "profiles.csv")
    comments   = pd.read_csv(DATA_DIR / "hate_comments.csv")
    engagement = pd.read_csv(Path("data") / "engagement.csv")  # shared across countries

    # ── Column validation ────────────────────────────────────────────────────
    required = {
        "profiles": {
            "profile_id", "topic", "age_group", "age", "gender", "origin",
            "username", "display_name", "avatar_initials", "avatar_colour",
            "target_message",
        },
        "comments":   {"topic", "severity", "ideology", "text"},
        "engagement": {"engagement_level", "likes", "comments_count"},
    }
    for name, df in [("profiles", profiles), ("comments", comments), ("engagement", engagement)]:
        missing = required[name] - set(df.columns)
        if missing:
            raise ValueError(f"[{name}.csv] Missing columns: {missing}")

    log.info(f"  profiles   : {len(profiles):>4} rows")
    log.info(f"  comments   : {len(comments):>4} rows")
    log.info(f"  engagement : {len(engagement):>4} rows")
    return profiles, comments, engagement


# ══════════════════════════════════════════════════════════════════════════════
# 2. BALANCED FACTORIAL DESIGN
# ══════════════════════════════════════════════════════════════════════════════

def generate_balanced_design(rng: np.random.Generator) -> pd.DataFrame:
    """
    Build the 18 000-row balanced design matrix.

    Strategy
    --------
    Non-topic factors: severity (3) × ideology (2) × age_group (2) ×
                       engagement_level (3) = 36 combinations.

    For each of the 6 topics:
      1. Tile the 36 combinations to fill N_RESPONDENTS slots (ceil replication).
      2. Shuffle the pool independently per topic using the shared RNG.

    Respondent i sees topic T at index i in topic T's shuffled pool.
    This guarantees:
      • Every respondent sees every topic exactly once.
      • Each combination occurs ≈ N_RESPONDENTS/36 ≈ 83–84 times per topic.
    """
    log.info("Generating balanced factorial design …")

    non_topic_combos = list(itertools.product(
        SEVERITIES, IDEOLOGIES, AGE_GROUPS, ENGAGEMENT_LEVELS  # 36 combos
    ))

    rows = []
    for topic in TOPICS:
        # Tile to N_RESPONDENTS, then trim to exact size
        n_reps = math.ceil(N_RESPONDENTS / len(non_topic_combos))
        pool   = (non_topic_combos * n_reps)[:N_RESPONDENTS]

        # Shuffle deterministically using the shared RNG
        idx  = rng.permutation(len(pool))
        pool = [pool[i] for i in idx]

        for resp_idx, (severity, ideology, age_group, eng_level) in enumerate(pool):
            rows.append({
                "respondent_id":   resp_idx + 1,
                "topic":           topic,
                "severity":        severity,
                "ideology":        ideology,
                "age_group":       age_group,
                "engagement_level": eng_level,
            })

    design_df = pd.DataFrame(rows)
    expected  = N_RESPONDENTS * N_VIGNETTES_PER_RESPONDENT
    log.info(f"  Design rows: {len(design_df):,}  (expected {expected:,})")
    return design_df


# ══════════════════════════════════════════════════════════════════════════════
# 3. CONTENT ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def assign_content(
    design_df:    pd.DataFrame,
    profiles_df:  pd.DataFrame,
    comments_df:  pd.DataFrame,
    engagement_df: pd.DataFrame,
    rng:          np.random.Generator,
) -> pd.DataFrame:
    """
    For every row in the design, randomly sample a compatible:
      • profile      – matched on (topic, age_group)
      • hate comment – matched on (topic, severity, ideology)
      • engagement   – matched on engagement_level
      • commenter username – drawn from the COMMENTER_USERNAMES pool

    Within-respondent constraint
    ----------------------------
    No profile_id may repeat inside the same respondent block.
    Because profiles are topic-specific this is satisfied automatically as
    long as each (topic, age_group) pool has at least one profile; a fallback
    lifts the constraint only if the pool is exhausted.
    """
    log.info("Assigning content to design cells …")

    # ── Build keyed lookup dicts for fast access ─────────────────────────────
    # Cross-ideological matching: conservative hate → progressive profiles and vice-versa.
    # Requires profiles.csv to have an 'ideology' column (added by prepare_data.py).
    has_profile_ideology = "ideology" in profiles_df.columns

    profile_idx: dict[tuple, list] = {}
    if has_profile_ideology:
        for (topic, age_group, ideology), grp in profiles_df.groupby(
            ["topic", "age_group", "ideology"]
        ):
            profile_idx[(topic, age_group, ideology)] = grp.to_dict("records")

        # Warn if any (topic, age_group, ideology) cell is missing — the pipeline
        # will fall back to same-ideology profiles for those cells.
        import itertools as _it
        missing_cells = [
            (t, a, i)
            for t, a, i in _it.product(
                profiles_df["topic"].unique(),
                profiles_df["age_group"].unique(),
                ["conservative", "progressive"],
            )
            if (t, a, i) not in profile_idx
        ]
        if missing_cells:
            log.warning(
                f"  {len(missing_cells)} profile cells missing ideology coverage "
                f"(run generate_conservative_profiles.py --merge to fix):"
            )
            for t, a, i in missing_cells[:6]:
                log.warning(f"    topic={t}, age_group={a}, ideology={i}")
    else:
        for (topic, age_group), grp in profiles_df.groupby(["topic", "age_group"]):
            profile_idx[(topic, age_group)] = grp.to_dict("records")

    comment_idx: dict[tuple, list] = {}
    for (topic, severity, ideology), grp in comments_df.groupby(
        ["topic", "severity", "ideology"]
    ):
        comment_idx[(topic, severity, ideology)] = grp.to_dict("records")

    engagement_idx: dict[str, list] = {}
    for level, grp in engagement_df.groupby("engagement_level"):
        engagement_idx[level] = grp.to_dict("records")

    # ── Iterate respondent by respondent ─────────────────────────────────────
    result_rows = []

    for resp_id, resp_df in design_df.groupby("respondent_id"):
        used_profile_ids: set = set()

        for _, row in resp_df.iterrows():
            topic     = row["topic"]
            severity  = row["severity"]
            ideology  = row["ideology"]
            age_group = row["age_group"]
            eng_level = row["engagement_level"]

            # ── Profile selection ────────────────────────────────────────────
            if has_profile_ideology:
                # Cross-ideological constraint: hate ideology determines which
                # profile ideology pool to draw from (opposites attract).
                profile_ideology = "progressive" if ideology == "conservative" else "conservative"
                key = (topic, age_group, profile_ideology)
            else:
                key = (topic, age_group)

            candidates = [
                p for p in profile_idx.get(key, [])
                if p["profile_id"] not in used_profile_ids
            ]
            # Fallback 1: allow reuse if all profiles for this cell are exhausted
            if not candidates:
                candidates = profile_idx.get(key, [])
            # Fallback 2: no profiles for this ideology yet — use any profile for
            # this (topic, age_group) regardless of ideology, with a warning.
            if not candidates and has_profile_ideology:
                fallback_key = next(
                    (k for k in profile_idx if k[0] == topic and k[1] == age_group),
                    None,
                )
                if fallback_key:
                    candidates = [
                        p for p in profile_idx[fallback_key]
                        if p["profile_id"] not in used_profile_ids
                    ] or profile_idx[fallback_key]
                    log.warning(
                        f"No {profile_ideology} profiles for topic={topic}, "
                        f"age_group={age_group} — using fallback ideology. "
                        f"Run generate_conservative_profiles.py --merge to fix."
                    )
            if not candidates:
                raise ValueError(
                    f"No profiles found for topic={topic}, age_group={age_group}"
                    + (f", ideology={profile_ideology}" if has_profile_ideology else "")
                )
            profile = candidates[rng.integers(len(candidates))]
            used_profile_ids.add(profile["profile_id"])

            # ── Comment selection ────────────────────────────────────────────
            c_pool = comment_idx.get((topic, severity, ideology), [])
            if not c_pool:
                raise ValueError(
                    f"No comments for topic={topic}, severity={severity}, ideology={ideology}"
                )
            comment = c_pool[rng.integers(len(c_pool))]

            # ── Engagement row selection ─────────────────────────────────────
            e_pool = engagement_idx.get(eng_level, [])
            if not e_pool:
                raise ValueError(f"No engagement rows for level={eng_level}")
            eng_row = e_pool[rng.integers(len(e_pool))]

            # ── Commenter username ───────────────────────────────────────────
            commenter = COMMENTER_USERNAMES[rng.integers(len(COMMENTER_USERNAMES))]

            # ── Post age (random from pool, seeded) ──────────────────────────
            post_age = POST_AGES[rng.integers(len(POST_AGES))]

            result_rows.append({
                **row.to_dict(),
                "profile_id":         profile["profile_id"],
                "profile_ideology":   profile.get("ideology", ""),
                "username":           profile["username"],
                "display_name":       profile["display_name"],
                "avatar_initials":    profile["avatar_initials"],
                "avatar_colour":      profile["avatar_colour"],
                "target_message":     profile["target_message"],
                "comment_text":       comment["text"],
                "commenter_username": commenter,
                "post_age":           post_age,
                "likes":              int(eng_row["likes"]),
                "comments_count":     int(eng_row["comments_count"]),
            })

    result_df = (
        pd.DataFrame(result_rows)
        .sort_values(["respondent_id", "topic"])
        .reset_index(drop=True)
    )

    # ── Assign randomised within-respondent vignette order ───────────────────
    all_orders: list[int] = []
    for _, grp in result_df.groupby("respondent_id"):
        order = rng.permutation(len(grp)) + 1
        all_orders.extend(order.tolist())

    result_df = result_df.sort_values("respondent_id").copy()
    result_df["vignette_order"] = all_orders
    result_df = result_df.sort_values(["respondent_id", "vignette_order"]).reset_index(drop=True)

    log.info(f"  Assigned {len(result_df):,} vignettes")
    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# 4. STIMULUS DEDUPLICATION & FILENAME ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def assign_stimulus_filenames(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify unique stimuli by their rendered content
    (profile_id, comment_text, likes, comments_count).

    Multiple respondents who draw the same combination share one stimulus file,
    saving render time without sacrificing stimulus variety.

    Adds columns: stimulus_id (int), stimulus_filename (str).
    """
    log.info("Deduplicating stimuli …")

    key_cols = ["profile_id", "comment_text", "likes", "comments_count"]
    unique   = metadata_df[key_cols].drop_duplicates().reset_index(drop=True)
    unique["stimulus_id"]       = range(1, len(unique) + 1)
    unique["stimulus_filename"] = unique["stimulus_id"].apply(
        lambda x: f"stimulus_{x:05d}.png"
    )

    metadata_df = metadata_df.merge(unique, on=key_cols, how="left")
    log.info(f"  Unique stimuli: {len(unique):,}  "
             f"(avg {len(metadata_df) / len(unique):.1f} respondents per stimulus)")
    return metadata_df


# ══════════════════════════════════════════════════════════════════════════════
# 5. HTML GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_html_stimuli(
    metadata_df: pd.DataFrame,
    env:         Environment,
    css_content: str,
) -> list[Path]:
    """
    Render one HTML file per unique stimulus using the Jinja2 template.

    The CSS is embedded inline so each HTML file is completely self-contained
    (no external asset dependencies when Playwright opens it via file://).

    Returns
    -------
    List of HTML file paths (one per unique stimulus).
    """
    log.info("Generating HTML stimuli …")

    template       = env.get_template("instagram_post.html")
    unique_stimuli = metadata_df.drop_duplicates("stimulus_id").copy()
    html_paths     = []
    _ui            = UI_STRINGS[COUNTRY]

    for _, row in unique_stimuli.iterrows():
        html_filename = row["stimulus_filename"].replace(".png", ".html")
        html_path     = HTML_DIR / html_filename

        _n = f"{int(row['comments_count']):,}"
        context = {
            "css":                css_content,
            "username":           row["username"],
            "display_name":       row["display_name"],
            "avatar_initials":    row["avatar_initials"],
            "avatar_colour":      row["avatar_colour"],
            "target_message":     row["target_message"],
            "commenter_username": row["commenter_username"],
            "comment_text":       row["comment_text"],
            "likes":              f"{int(row['likes']):,}",
            "comments_count":     _n,
            "topic_gradient":     TOPIC_GRADIENTS.get(
                                      row["topic"],
                                      "linear-gradient(135deg, #667eea, #764ba2)"
                                  ),
            "topic_emoji":        TOPIC_EMOJIS.get(row["topic"], "📱"),
            "post_age":           row["post_age"],
            # i18n UI strings
            "lang":        COUNTRY,
            "ui_likes":    f"{int(row['likes']):,} {_ui['likes']}",
            "ui_view_all": _ui["view_all"].format(n=_n),
            "ui_view":     _ui["view"].format(n=_n),
            "ui_more":     _ui["more"],
        }

        html_path.write_text(template.render(**context), encoding="utf-8")
        html_paths.append(html_path)

    log.info(f"  Generated {len(html_paths):,} HTML files → {HTML_DIR}")
    return html_paths


# ══════════════════════════════════════════════════════════════════════════════
# 6. SCREENSHOT RENDERING
# ══════════════════════════════════════════════════════════════════════════════

async def take_screenshots(
    html_paths: list,
    limit: Optional[int] = None,
) -> None:
    """
    Render HTML files to PNG using Playwright Chromium.

    Key settings
    ------------
    • Viewport  : 375 × 812 px (standard iPhone form-factor)
    • Scale     : deviceScaleFactor=2 (Retina / HiDPI quality)
    • Target    : '.ig-wrapper' element (crops to post card only)
    • Batching  : SCREENSHOT_BATCH_SIZE concurrent pages to avoid OOM

    Skips files already rendered (idempotent re-runs).
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        log.error(
            "Playwright is not installed.\n"
            "Run: pip install playwright && playwright install chromium"
        )
        return

    targets = html_paths[:limit] if limit else html_paths
    total   = len(targets)
    done    = 0

    log.info(f"Rendering {total:,} screenshots …  (batch={SCREENSHOT_BATCH_SIZE})")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()

        for batch_start in range(0, total, SCREENSHOT_BATCH_SIZE):
            batch = targets[batch_start : batch_start + SCREENSHOT_BATCH_SIZE]

            async def render_one(html_path: Path) -> None:
                png_path = PNG_DIR / html_path.name.replace(".html", ".png")
                if png_path.exists():
                    return  # already rendered — skip

                page = await browser.new_page(
                    viewport={"width": 375, "height": 812},
                    device_scale_factor=2,
                )
                try:
                    await page.goto(
                        f"file://{html_path.resolve()}",
                        wait_until="networkidle",
                    )
                    # Screenshot only the post wrapper (crops chrome chrome)
                    element = await page.query_selector(".ig-wrapper")
                    if element:
                        await element.screenshot(path=str(png_path))
                    else:
                        await page.screenshot(path=str(png_path), full_page=False)
                finally:
                    await page.close()

            await asyncio.gather(*[render_one(p) for p in batch])
            done += len(batch)
            log.info(f"  {done}/{total} rendered")

        await browser.close()

    log.info(f"Screenshots saved → {PNG_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Instagram-style hate-speech vignette stimuli."
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of PNG screenshots rendered (useful for testing)"
    )
    parser.add_argument(
        "--skip-screenshots", action="store_true",
        help="Generate metadata + HTML only; skip Playwright rendering"
    )
    parser.add_argument(
        "--country", default="en",
        choices=["en", "it", "es", "fr", "de"],
        help="Country code for localised data and output (default: en)"
    )
    args = parser.parse_args()

    # ── Apply country-specific path and string overrides ─────────────────────
    global COUNTRY, DATA_DIR, OUTPUT_DIR, HTML_DIR, PNG_DIR, METADATA_DIR, POST_AGES
    COUNTRY = args.country
    if COUNTRY != "en":
        DATA_DIR     = Path("data/countries") / COUNTRY
        OUTPUT_DIR   = Path("output") / COUNTRY
        HTML_DIR     = OUTPUT_DIR / "html"
        PNG_DIR      = OUTPUT_DIR / "png"
        METADATA_DIR = OUTPUT_DIR / "metadata"
    POST_AGES = UI_STRINGS[COUNTRY]["post_ages"]

    rng = np.random.default_rng(args.seed)

    # ── Create output directories ─────────────────────────────────────────────
    for d in (HTML_DIR, PNG_DIR, METADATA_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1 : load data ────────────────────────────────────────────────────
    profiles_df, comments_df, engagement_df = load_data()

    # ── Step 2 : generate balanced design ────────────────────────────────────
    design_df = generate_balanced_design(rng)

    # ── Step 3 : assign content per cell ──────────────────────────────────────
    metadata_df = assign_content(
        design_df, profiles_df, comments_df, engagement_df, rng
    )

    # ── Step 4 : deduplicate and name stimulus files ───────────────────────────
    metadata_df = assign_stimulus_filenames(metadata_df)

    # ── Step 5 : save metadata CSV ────────────────────────────────────────────
    output_cols = [
        "respondent_id", "vignette_order", "topic", "severity", "ideology",
        "age_group", "engagement_level", "profile_id", "comment_text",
        "likes", "comments_count", "stimulus_filename",
    ]
    out_path = METADATA_DIR / "vignette_metadata.csv"
    metadata_df[output_cols].to_csv(out_path, index=False)
    log.info(f"Metadata saved  → {out_path}  ({len(metadata_df):,} rows)")

    # ── Step 6 : generate HTML stimuli ───────────────────────────────────────
    css_content = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
    env         = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    html_paths  = generate_html_stimuli(metadata_df, env, css_content)

    # ── Step 7 : render PNG screenshots ───────────────────────────────────────
    if args.skip_screenshots:
        log.info("Skipping screenshot rendering (--skip-screenshots)")
    else:
        asyncio.run(take_screenshots(html_paths, limit=args.limit))

    log.info("Pipeline complete ✓")


if __name__ == "__main__":
    main()
