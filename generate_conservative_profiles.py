#!/usr/bin/env python3
"""
generate_conservative_profiles.py
==================================
Generates conservative/majority-group profiles for the five topics that lack
them (all topics except Nationalism, which already has mixed-ideology profiles).

For each topic, generates 12 profiles (6 young_adult + 6 adolescent) representing
majority-group members posting culturally conservative content — not hateful,
but reflecting a conservative worldview that would naturally receive progressive
hate comments as a response.

After generation, review the output in visualise_inputs.py, then merge into
data/profiles.csv by running:

    python generate_conservative_profiles.py --merge

Usage
-----
    python generate_conservative_profiles.py              # generate only
    python generate_conservative_profiles.py --merge      # generate + merge
    python generate_conservative_profiles.py --merge --skip-generate  # merge only
    python generate_conservative_profiles.py --dry-run   # print prompts, no API calls
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

MODEL      = "claude-opus-4-7"
MAX_TOKENS = 8192

DATA_DIR   = pathlib.Path("data")
OUTPUT_CSV = DATA_DIR / "conservative_profiles.csv"

PROFILE_FIELDS = [
    "profile_id", "topic", "age_group", "age", "gender", "origin",
    "username", "display_name", "avatar_initials", "avatar_colour",
    "target_message", "ideology", "identity_tags",
]

# ──────────────────────────────────────────────────────────────────────────────
# Starting profile_id after the existing 72 progressive profiles
# (Nationalism already has 5 conservative profiles P031, P034, P067, P069, P071
#  encoded via prepare_data.py — no new ones needed for that topic)
START_ID = 73

# Avatar colour palette (cycles across generated profiles)
AVATAR_COLOURS = [
    "#E91E8C", "#2196F3", "#9C27B0", "#FF5722",
    "#4CAF50", "#FF9800", "#00BCD4", "#F44336",
    "#3F51B5", "#009688", "#8BC34A", "#FFC107",
]

# ──────────────────────────────────────────────────────────────────────────────
# Topic specifications for conservative profile generation
# ──────────────────────────────────────────────────────────────────────────────

TOPIC_SPECS: dict[str, dict] = {
    "Immigration / migrants": {
        "persona":  "white European or American who is concerned about immigration levels and cultural change, "
                    "posting from a 'put citizens first' or cultural-preservation perspective",
        "example_messages": [
            "Our country's resources belong to our people first. Open borders are not compassion — they are chaos. #NationFirst 🏴",
            "Proud of my culture and determined to protect it. We can be welcoming AND have standards. Nothing extreme about that. 🇩🇪",
            "I don't hate immigrants — I just believe my country has the right to decide who comes and how many. That's called sovereignty. 🇫🇷",
        ],
        "identity_tags": "nativist",
    },
    "Religion (Muslim / Jewish)": {
        "persona":  "devout Christian or religious traditionalist who posts about faith, traditional family values, "
                    "and the importance of religious heritage in society",
        "example_messages": [
            "My faith is the foundation of everything I do. Traditional values built this civilisation and will sustain it. 🙏 #ChristianLife",
            "Faith family and community — the pillars every society needs. Proud to live by the values I was raised with. ✝️",
            "Practising my faith openly and unapologetically. My beliefs are not a threat — they are who I am. 🙏 #TradValues",
        ],
        "identity_tags": "religious_conservative",
    },
    "Sexual orientation / gender identity": {
        "persona":  "religious conservative or traditional-values holder who posts about protecting children, "
                    "traditional family structures, or biological sex — without being overtly hateful but clearly "
                    "opposed to LGBTQ+ ideology in schools or public policy",
        "example_messages": [
            "Proud of my traditional family — mum dad and kids. This model works and it is worth protecting. #FamilyValues 🏡",
            "Children deserve to grow up without adult political agendas in the classroom. Parental rights matter. 🙏",
            "Biological sex is real science. I will keep saying it no matter how unpopular it becomes. #SpeakingTruth",
        ],
        "identity_tags": "anti_LGBTQ",
    },
    "Gender issues (misogyny)": {
        "persona":  "anti-feminist man or traditionalist woman who posts about men's rights, "
                    "the 'feminism has gone too far' narrative, or traditional gender roles — "
                    "not overtly misogynistic but clearly opposing feminist activism",
        "example_messages": [
            "Feminism was once about equality. Now it is about power at the expense of men and boys. That has to be said. 💪",
            "Traditional femininity is not oppression — it is a choice many women freely make. Stop erasing us. 🌸 #TradWife",
            "Men are falling behind in education employment and mental health and nobody is allowed to talk about it. That changes now. 💪",
        ],
        "identity_tags": "traditional_man",
    },
    "Racism / ethnicity": {
        "persona":  "white European or white American who posts about pride in European heritage, "
                    "Western civilisation, or the idea that cultural preservation is not racism — "
                    "using 'heritage pride' framing without explicit racial slurs",
        "example_messages": [
            "Proud of my European heritage and the civilisation my ancestors built. Defending it is not hate — it is pride. 🇩🇪🏛️",
            "Western civilisation produced science democracy art and medicine. I am not ashamed to say I value what my culture created. 🇫🇷",
            "You are allowed to be proud of every heritage EXCEPT European. I reject that double standard entirely. #EuropeanPride 🇪🇺",
        ],
        "identity_tags": "white_European_nationalist",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

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
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))
    return json.loads(text)


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


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt(topic: str, age_group: str, spec: dict, profiles_batch: list[dict]) -> str:
    ages = "18–25" if age_group == "young_adult" else "13–17"
    age_label = "young adult" if age_group == "young_adult" else "teenager"
    examples = "\n".join(f"  - \"{m}\"" for m in spec["example_messages"])

    return f"""You are a social-science researcher designing Instagram-style survey stimuli for a study on online hate speech.

**Task**: Generate {len(profiles_batch)} realistic Instagram profile records for the topic "{topic}".

**Persona type**: {spec['persona']}

**Age group**: {age_group} ({ages} years old, so the poster is a {age_label})

**Constraints**:
- Each profile must feel like a real young person on Instagram, not an extremist caricature.
- The `target_message` is a single Instagram post caption (1–3 sentences, ≤280 characters).
  It expresses a conservative viewpoint naturally — like something you would genuinely see on Instagram.
  It must NOT contain slurs, explicit dehumanisation, or direct calls to violence.
  It CAN be provocative or politically charged, but plausibly deniable as opinion.
- Example `target_message` styles for this topic:
{examples}
- Mix of male/female/non-binary genders appropriate for the persona.
- Origins: mostly white European or white North American. Use a mix of nationalities (British, French, German, Italian, Spanish, American, etc.) appropriate to the persona.
- Usernames: realistic Instagram handles (no spaces; underscores/dots allowed). Suggest the conservative identity subtly in the username.
- Display names: realistic Western names matching the origin.
- avatar_initials: first letter of first name + first letter of surname.
- avatar_colour: use one of these hex colours (vary across profiles): #E91E8C #2196F3 #9C27B0 #FF5722 #4CAF50 #FF9800 #00BCD4 #F44336 #3F51B5 #009688 #8BC34A #FFC107
- ideology: always "conservative"
- identity_tags: always "{spec['identity_tags']}"

**Profiles to generate** (provide these exact profile_id values):
{json.dumps([p["profile_id"] for p in profiles_batch])}

Return a JSON array of {len(profiles_batch)} objects. Each object MUST have exactly these keys:
profile_id, topic, age_group, age, gender, origin, username, display_name,
avatar_initials, avatar_colour, target_message, ideology, identity_tags

No explanations — return ONLY the JSON array."""


def generate_profiles_for_topic(
    topic: str,
    spec: dict,
    start_id: int,
    client: anthropic.Anthropic,
    dry_run: bool = False,
) -> list[dict]:
    age_groups = [
        ("young_adult", [start_id + i for i in range(6)]),
        ("adolescent",  [start_id + 6 + i for i in range(6)]),
    ]

    all_profiles = []
    for age_group, ids in age_groups:
        batch_shells = [
            {"profile_id": f"P{pid:03d}", "topic": topic, "age_group": age_group}
            for pid in ids
        ]
        log.info(f"  {topic[:30]:30}  {age_group}  (P{ids[0]:03d}–P{ids[-1]:03d})")

        prompt = build_prompt(topic, age_group, spec, batch_shells)

        if dry_run:
            print(f"\n{'─'*70}\nPROMPT for {topic} / {age_group}:\n{prompt}\n")
            all_profiles.extend(batch_shells)
            continue

        raw     = call_claude(client, prompt)
        results = parse_json_response(raw)

        if len(results) != len(batch_shells):
            log.warning(f"  Got {len(results)} profiles, expected {len(batch_shells)}")

        id_map = {s["profile_id"]: s for s in batch_shells}
        for r in results:
            pid  = r.get("profile_id")
            base = id_map.get(pid, {})
            merged = {**base, **r}
            merged.setdefault("ideology",      "conservative")
            merged.setdefault("identity_tags", spec["identity_tags"])
            all_profiles.append(merged)

    return all_profiles


def generate_all(dry_run: bool = False) -> list[dict]:
    client  = anthropic.Anthropic() if not dry_run else None
    all_new = []
    current_id = START_ID

    for topic, spec in TOPIC_SPECS.items():
        log.info(f"Generating conservative profiles: {topic}")
        profiles = generate_profiles_for_topic(
            topic=topic,
            spec=spec,
            start_id=current_id,
            client=client,
            dry_run=dry_run,
        )
        all_new.extend(profiles)
        current_id += 12   # 12 profiles per topic (6 young_adult + 6 adolescent)

    return all_new


# ──────────────────────────────────────────────────────────────────────────────
# Merge
# ──────────────────────────────────────────────────────────────────────────────

def merge_into_profiles(conservative_csv: pathlib.Path, profiles_csv: pathlib.Path) -> None:
    existing = read_csv(profiles_csv)
    new_rows  = read_csv(conservative_csv)

    existing_ids = {r["profile_id"] for r in existing}
    to_add = [r for r in new_rows if r["profile_id"] not in existing_ids]
    skipped = len(new_rows) - len(to_add)

    if skipped:
        log.info(f"  Skipping {skipped} profile_ids already present in profiles.csv")

    merged = existing + to_add

    # Ensure ideology and identity_tags columns are present in fieldnames
    fieldnames = list(existing[0].keys()) if existing else PROFILE_FIELDS
    for col in ("ideology", "identity_tags"):
        if col not in fieldnames:
            fieldnames.append(col)

    write_csv(profiles_csv, merged, fieldnames)
    log.info(f"  Merged {len(to_add)} new profiles → {profiles_csv} ({len(merged)} total)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate conservative profiles and optionally merge into profiles.csv."
    )
    parser.add_argument("--merge",           action="store_true",
                        help="After generation, merge into data/profiles.csv")
    parser.add_argument("--skip-generate",   action="store_true",
                        help="Skip generation; only merge existing conservative_profiles.csv")
    parser.add_argument("--dry-run",         action="store_true",
                        help="Print prompts but make no API calls")
    parser.add_argument("--output",          type=pathlib.Path, default=OUTPUT_CSV,
                        help=f"Output CSV path (default: {OUTPUT_CSV})")
    args = parser.parse_args()

    if not args.skip_generate:
        log.info("Generating conservative profiles …")
        profiles = generate_all(dry_run=args.dry_run)

        if not args.dry_run:
            write_csv(args.output, profiles, PROFILE_FIELDS)
            log.info(f"Generated {len(profiles)} conservative profiles → {args.output}")
            log.info("Review the output in visualise_inputs.py before merging.")
        else:
            log.info(f"[dry-run] Would write {len(profiles)} profiles to {args.output}")

    if args.merge and not args.dry_run:
        if not args.output.exists():
            log.error(f"Cannot merge: {args.output} not found. Run without --skip-generate first.")
            sys.exit(1)
        profiles_csv = DATA_DIR / "profiles.csv"
        if not profiles_csv.exists():
            log.error(f"Cannot merge: {profiles_csv} not found.")
            sys.exit(1)
        merge_into_profiles(args.output, profiles_csv)


if __name__ == "__main__":
    main()
