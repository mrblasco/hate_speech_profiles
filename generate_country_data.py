#!/usr/bin/env python3
"""
generate_country_data.py
========================
Translate and culturally adapt the English stimulus content (profiles + hate
comments) into one of four EU country languages using the Claude API.

Run once per target country to populate data/countries/{cc}/.

Usage
-----
    python generate_country_data.py --country it   # Italy
    python generate_country_data.py --country es   # Spain
    python generate_country_data.py --country fr   # France
    python generate_country_data.py --country de   # Germany
    python generate_country_data.py --country it --only profiles
    python generate_country_data.py --country it --only comments
"""

import argparse
import csv
import json
import logging
import os
import pathlib
import sys
import time

import anthropic

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────

COUNTRY_META = {
    "it": {
        "name":     "Italy",
        "language": "Italian",
        "minority_communities": (
            "Roma/Sinti, sub-Saharan African migrants, North African (Maghrebi) Muslims, "
            "Eastern European workers (Romanian, Albanian), Chinese, LGBTQ+ people, "
            "Jewish communities, refugees and asylum seekers"
        ),
        "platforms": "Instagram, TikTok, Facebook, Twitter/X, Telegram",
        "political_context": (
            "Far-right parties (Fratelli d'Italia, Lega) frame immigration as an "
            "existential threat; anti-Islam sentiment; strong Catholic identity politics; "
            "anti-Roma hostility; antisemitism on both far-right and far-left"
        ),
    },
    "es": {
        "name":     "Spain",
        "language": "Spanish",
        "minority_communities": (
            "Maghrebi (Moroccan) Muslims, Latin American immigrants, Roma/Gitano, "
            "sub-Saharan African migrants, Chinese, LGBTQ+ people, Jewish communities, "
            "Venezuelan/Colombian refugees"
        ),
        "platforms": "Instagram, TikTok, Twitter/X, Facebook, WhatsApp",
        "political_context": (
            "Far-right Vox party frames immigration from Africa as invasion; "
            "anti-Islam sentiment especially regarding Moroccan immigration; "
            "anti-Roma prejudice; Spanish nationalist identity vs. regional separatism; "
            "anti-LGBTQ+ from religious right"
        ),
    },
    "fr": {
        "name":     "France",
        "language": "French",
        "minority_communities": (
            "Maghrebi (Algerian, Moroccan, Tunisian) Muslims, sub-Saharan Africans, "
            "Roma, Black French people, Jewish communities, LGBTQ+ people, "
            "refugees and sans-papiers"
        ),
        "platforms": "Instagram, TikTok, Twitter/X, Facebook, Snapchat",
        "political_context": (
            "Rassemblement National (Marine Le Pen, Jordan Bardella) frames Muslim "
            "immigration as incompatible with laïcité and French values; "
            "anti-Semitic incidents from both far-right and Islamist milieux; "
            "anti-Roma hostility; intense debate around secularism (laïcité) and Islam; "
            "anti-Black racism and colonial legacy"
        ),
    },
    "de": {
        "name":     "Germany",
        "language": "German",
        "minority_communities": (
            "Turkish-German Muslims, Syrian and Afghan refugees, Roma/Sinti, "
            "sub-Saharan African migrants, Jewish communities, LGBTQ+ people, "
            "Eastern European workers (Polish, Romanian)"
        ),
        "platforms": "Instagram, TikTok, Twitter/X, Facebook, YouTube",
        "political_context": (
            "AfD frames migration as Überfremdung (over-foreignisation); "
            "anti-Islam sentiment especially targeting Turkish and Arab communities; "
            "antisemitism from far-right and some Muslim milieus; "
            "anti-Roma discrimination; trans/queer hostility on conservative right; "
            "Islamist extremism as counter-narrative"
        ),
    },
}

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")   # ← REPLACE with your actual API key
MODEL = "claude-opus-4-7"
MAX_TOKENS = 8192

# ──────────────────────────────────────────────────────────────────────────────


def read_csv(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"  Wrote {len(rows)} rows → {path}")


def call_claude(client: anthropic.Anthropic, prompt: str) -> str:
    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as exc:
            log.warning(f"API error (attempt {attempt + 1}/3): {exc}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError("Claude API call failed after 3 attempts")


def parse_json_response(text: str) -> list[dict]:
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.startswith("```")
        )
    return json.loads(text)


# ──────────────────────────────────────────────────────────────────────────────
# Profiles
# ──────────────────────────────────────────────────────────────────────────────

PROFILE_KEEP_UNCHANGED = {
    "profile_id", "age_group", "age", "gender", "topic",
    "avatar_colour", "avatar_initials",
}

PROFILE_TRANSLATE_FIELDS = {
    "origin", "username", "display_name", "target_message",
}

PROFILE_BATCH_SIZE = 12   # rows per API call


def adapt_profiles(
    rows: list[dict],
    meta: dict,
    client: anthropic.Anthropic,
) -> list[dict]:
    country, language = meta["name"], meta["language"]
    communities       = meta["minority_communities"]
    context           = meta["political_context"]
    platforms         = meta["platforms"]

    adapted = []
    total   = len(rows)

    for i in range(0, total, PROFILE_BATCH_SIZE):
        batch = rows[i : i + PROFILE_BATCH_SIZE]
        log.info(f"  Adapting profiles {i+1}–{min(i+len(batch), total)} / {total} …")

        # Strip unchanged fields so the prompt stays lean
        slim = [
            {k: v for k, v in r.items() if k in PROFILE_TRANSLATE_FIELDS or k == "profile_id"}
            for r in batch
        ]

        prompt = f"""You are a social-science researcher adapting Instagram-style survey stimuli for a study on online hate speech in {country}.

**Task**: Translate and culturally adapt the following Instagram profile data rows into {language}.

**Instructions**:
- Translate ALL text fields into {language} (natural, idiomatic social-media language).
- `username`: create a realistic {language}-language Instagram username (no spaces, use underscores or dots; keep it plausible for a young person in {country}).
- `display_name`: realistic first name + surname for {country} (no translation needed for names — use authentic {country} names).
- `origin`: replace with a plausible origin/background for a young person in {country}; may reference minority communities: {communities}.
- `target_message`: this is the text shown on the Instagram post image. Translate it into {language}; keep the same general meaning, but adapt cultural references to {country} context, referencing locally relevant communities where appropriate.
- Preserve `profile_id` EXACTLY as-is.
- Do NOT include fields that are not in the input rows.

**{country} socio-political context** (for cultural adaptation of `target_message`):
{context}

**Common platforms in {country}**: {platforms}

**Input rows** (JSON array):
{json.dumps(slim, ensure_ascii=False, indent=2)}

Return ONLY a valid JSON array with the same structure and `profile_id` values. No explanations."""

        raw     = call_claude(client, prompt)
        results = parse_json_response(raw)

        # Merge translated fields back into original rows (preserve unchanged cols)
        id_to_original = {r["profile_id"]: r for r in batch}
        for res in results:
            pid  = res["profile_id"]
            orig = id_to_original.get(pid)
            if orig is None:
                log.warning(f"  Unexpected profile_id {pid} in response — skipping")
                continue
            merged = {**orig}
            for field in PROFILE_TRANSLATE_FIELDS:
                if field in res:
                    merged[field] = res[field]
            adapted.append(merged)

    return adapted


# ──────────────────────────────────────────────────────────────────────────────
# Hate comments
# ──────────────────────────────────────────────────────────────────────────────

COMMENT_BATCH_SIZE = 18   # rows per API call


def adapt_comments(
    rows: list[dict],
    meta: dict,
    client: anthropic.Anthropic,
) -> list[dict]:
    country, language = meta["name"], meta["language"]
    communities       = meta["minority_communities"]
    context           = meta["political_context"]

    adapted = []
    total   = len(rows)

    for i in range(0, total, COMMENT_BATCH_SIZE):
        batch = rows[i : i + COMMENT_BATCH_SIZE]
        log.info(f"  Adapting comments {i+1}–{min(i+len(batch), total)} / {total} …")

        prompt = f"""You are a social-science researcher adapting hate-speech survey stimuli for a study on online hate in {country}.

**Task**: Translate and culturally adapt the following hate comments into {language}.

**CRITICAL rules** (this is academic stimulus material — fidelity to severity is essential):
1. Translate each comment into {language} using authentic social-media vernacular for {country}.
2. Preserve the EXACT severity level:
   - `opinion`: expresses prejudiced opinion or stereotype, no dehumanisation
   - `dehumanising`: removes humanity, uses animal/pest/disease metaphors, explicit dehumanisation
   - `incitement`: calls for violence, expulsion, or mass harm; most extreme
3. Replace minority group targets with the most locally salient equivalent for {country}:
   {communities}
   Map English targets to {country} equivalents (e.g. "immigrants" in immigration topic → Moroccan or sub-Saharan migrants as locally relevant; "Muslims" → Turkish/Arab Muslims as appropriate).
4. Keep conservative vs progressive ideological framing intact.
5. Keep `topic`, `severity`, `ideology` columns UNCHANGED.
6. Return the same number of rows as input.

**{country} socio-political context**:
{context}

**Input rows** (JSON array with fields: topic, severity, ideology, text):
{json.dumps(batch, ensure_ascii=False, indent=2)}

Return ONLY a valid JSON array. Each object must have keys: topic, severity, ideology, text. No explanations."""

        raw     = call_claude(client, prompt)
        results = parse_json_response(raw)

        for orig, res in zip(batch, results):
            adapted.append({
                "topic":    orig["topic"],
                "severity": orig["severity"],
                "ideology": orig["ideology"],
                "text":     res["text"],
            })

    return adapted


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate and adapt survey stimuli to a target EU country."
    )
    parser.add_argument(
        "--country", required=True,
        choices=list(COUNTRY_META.keys()),
        help="Target country code (it, es, fr, de)"
    )
    parser.add_argument(
        "--only", choices=["profiles", "comments"],
        help="Run only one adaptation step (default: both)"
    )
    args = parser.parse_args()

    cc   = args.country
    meta = COUNTRY_META[cc]
    log.info(f"Adapting stimuli for {meta['name']} ({meta['language']}) …")

    data_root = pathlib.Path("data")
    out_dir   = data_root / "countries" / cc
    client    = anthropic.Anthropic()

    # ── Profiles ──────────────────────────────────────────────────────────────
    if args.only in (None, "profiles"):
        profiles_src = read_csv(data_root / "profiles.csv")
        log.info(f"Loaded {len(profiles_src)} English profiles")
        adapted_profiles = adapt_profiles(profiles_src, meta, client)

        if len(adapted_profiles) != len(profiles_src):
            log.warning(
                f"Profile count mismatch: got {len(adapted_profiles)}, "
                f"expected {len(profiles_src)}"
            )

        write_csv(
            out_dir / "profiles.csv",
            adapted_profiles,
            fieldnames=list(profiles_src[0].keys()),
        )

    # ── Hate comments ─────────────────────────────────────────────────────────
    if args.only in (None, "comments"):
        comments_src = read_csv(data_root / "hate_comments.csv")
        log.info(f"Loaded {len(comments_src)} English hate comments")
        adapted_comments = adapt_comments(comments_src, meta, client)

        if len(adapted_comments) != len(comments_src):
            log.warning(
                f"Comment count mismatch: got {len(adapted_comments)}, "
                f"expected {len(comments_src)}"
            )

        write_csv(
            out_dir / "hate_comments.csv",
            adapted_comments,
            fieldnames=["topic", "severity", "ideology", "text"],
        )

    log.info(f"Done. Files written to {out_dir}/")


if __name__ == "__main__":
    main()
