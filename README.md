# Instagram Vignette Stimulus Generator

Reproducible Python pipeline for generating synthetic Instagram vignette stimuli
for academic research on online hate speech escalation.

---

## Project structure

```
.
├── configs/
│   ├── study_config.yaml        # Design parameters, generation settings
│   ├── prompts.yaml             # All LLM prompts (externally versioned)
│   └── generation_rules.yaml   # Content rules and validation thresholds
│
├── src/
│   ├── main.py                  # CLI entry point
│   ├── models.py                # Pydantic schemas (Profile, Post, Comment, …)
│   ├── sampling.py              # Balanced factorial design matrix
│   ├── prompts.py               # Prompt builder (loads from prompts.yaml)
│   ├── llm_client.py            # Async OpenAI-compatible client w/ retry & cache
│   ├── generators/
│   │   ├── profiles.py          # Profile generation
│   │   ├── posts.py             # Post (caption) generation
│   │   └── comments.py         # Comment generation (all 3 severities)
│   ├── validators/
│   │   ├── schema.py            # Rule-based + fuzzy dedup validation
│   │   ├── severity.py          # LLM judge for severity classification
│   │   └── realism.py          # LLM realism check for posts
│   ├── pipelines/
│   │   └── generation_pipeline.py  # Full pipeline orchestration
│   └── utils/
│       ├── io.py                # JSON / CSV / YAML / disk cache helpers
│       ├── hashing.py           # SHA-256 utilities
│       ├── seeds.py             # Deterministic seed derivation
│       └── logging_utils.py     # Logging setup
│
├── scripts/                     # Existing HTML/PNG rendering pipeline
│   ├── pipeline.py
│   ├── generate_conservative_profiles.py
│   └── …
│
├── data/                        # Input CSVs (profiles, comments, engagement)
├── outputs/                     # Generated stimuli (CSV, JSONL, metadata)
├── .env.example                 # Environment variable template
├── requirements.txt
└── Makefile
```

---

## Installation

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium   # only needed for HTML→PNG rendering
```

## API setup

Copy `.env.example` to `.env` and add your credentials:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

The pipeline uses any **OpenAI-compatible** endpoint. To use a different
provider, set `OPENAI_BASE_URL` in `.env`:

```bash
# Together AI
OPENAI_BASE_URL=https://api.together.xyz/v1
OPENAI_MODEL=meta-llama/Llama-3-70b-chat-hf

# Local Ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3
```

---

## Running the LLM generation pipeline

```bash
# Default run: 50 profiles, seed 42, outputs to outputs/run_001/
python src/main.py

# Full options
python src/main.py \
  --n_profiles 50 \
  --seed 42 \
  --output_dir outputs/run_001 \
  --model gpt-4o

# Dry run (design matrix only, no API calls)
python src/main.py --n_profiles 50 --dry-run

# Skip validation judges (faster, less rigorous)
python src/main.py --n_profiles 50 --no-judge --no-realism

# Via Makefile
make generate          # 50 profiles, seed 42
make generate-dry      # dry run
make generate-fast     # no validation judges
```

---

## Running the HTML/PNG rendering pipeline

```bash
make pipeline                   # English, all screenshots
make pipeline-html              # English, HTML only
make stimuli COUNTRY=it         # Italian localization
make country COUNTRY=de         # Adapt + render (German)
```

---

## Pipeline structure

```
Conditions ──► Profiles ──► Posts ──► Comments (×3 severities)
                 │            │            │
               Schema       Schema      Schema
               check        check       check
                             │            │
                           Realism    Severity
                           judge       judge
                             │            │
                         ───────────────────
                                 │
                           StimulusRows
                                 │
                    ┌────────────┴────────────┐
                 stimuli.csv           stimuli.jsonl
                             generation_metadata.json
```

### Stage descriptions

| Stage | What happens |
|-------|-------------|
| 1. Sampling | Balanced factorial design matrix: all combinations of topic × age_group × gender × values are tiled and shuffled deterministically. |
| 2. Profiles | One LLM call per profile; Pydantic schema validation. |
| 3. Posts | One LLM call per profile; word count and forbidden-pattern checks. |
| 4. Post validation | Rule-based schema check + optional LLM realism judge. |
| 5. Comments | Three LLM calls per post (one per severity level), all concurrent. |
| 6. Comment validation | Rule-based schema check + optional LLM severity judge. Items where judge disagrees with intended severity are flagged and optionally rejected. |
| 7. Assembly | Fully denormalised `StimulusRow` records (one row = one stimulus cell). |
| 8. Output | CSV, JSONL, and full metadata JSON written to `outputs/<run>/final/`. |

---

## Experimental design

| Factor | Levels |
|--------|--------|
| Topic | immigration, feminism, religion, climate, public_health, national_identity |
| Age group | adolescent (13–17), young_adult (18–25) |
| Gender | male, female, nonbinary |
| Values | progressive, conservative |
| Comment severity | opposing_opinion, dehumanising, inciting_violence |

Every profile generates **one post** and **three comments** (one per severity).
The post is fixed across severity conditions — only the comment is the manipulation.

---

## Reproducibility guarantees

Every generated item stores:

- **prompt_text** — the exact prompt sent to the LLM
- **prompt_hash** — SHA-256 of the prompt (16-char prefix)
- **model_name** — exact model identifier
- **temperature** — generation temperature
- **seed** — per-item derived seed
- **timestamp** — UTC timestamp
- **run_id** — unique hex ID for the run
- **experiment_id** — hash of (seed, n_profiles, model)

Seeds are derived deterministically:
```
item_seed = SHA-256(base_seed + "|" + item_type + "|" + item_id)
```

This means each item has a unique seed without a shared stateful RNG,
making individual items reproducible in isolation.

The `generation_metadata.json` file includes a SHA-256 hash of
`study_config.yaml` at run time, so any change to the config is detectable.

---

## Output datasets

### `outputs/<run>/final/stimuli.csv`

One row per stimulus (post × severity). Key columns:

| Column | Description |
|--------|-------------|
| `stimulus_id` | `{post_id}_{severity}` |
| `profile_id` | Profile identifier |
| `topic` | Experimental topic |
| `age_group` | Profile age group |
| `gender` | Profile gender |
| `values` | Profile values (progressive/conservative) |
| `caption` | Original Instagram post (never contains hate speech) |
| `severity` | Comment severity condition |
| `comment_text` | Generated comment |
| `toxicity_estimate` | Self-reported LLM toxicity estimate (0–1) |
| `judge_severity_score` | Independent judge score (1–3) |
| `judge_agrees` | Whether judge agrees with intended severity |
| `realism_score` | Post realism score (0–1) |
| `passed_validation` | Whether item passed all validation checks |
| `model_name` | LLM used |
| `prompt_hash` | Prompt provenance hash |
| `seed` | Derivation seed |
| `timestamp` | Generation timestamp |

### `outputs/<run>/final/stimuli.jsonl`

Same content as CSV, one JSON object per line. Easier to load in Python:

```python
import pandas as pd
df = pd.read_json("outputs/run_001/final/stimuli.jsonl", lines=True)
```

### `outputs/<run>/final/generation_metadata.json`

Run-level manifest with experiment ID, timestamps, counts, model, config hash,
and file paths.

### `outputs/<run>/raw/`

Intermediate JSON artifacts saved after each generation stage, before validation.
Useful for debugging failed runs without regenerating everything.

### `outputs/<run>/validated/stimuli_all.jsonl`

All stimulus rows including those that failed validation (`passed_validation: false`).

---

## Validation strategy

### A. Schema validation (Pydantic)

All generated objects are parsed through strict Pydantic models immediately
after LLM generation. Invalid JSON or missing required fields raise errors
that are caught and logged per item.

### B. Rule-based content validation

Posts are checked for:
- Word count (15–40 words)
- Forbidden patterns (slurs and explicit violence vocabulary from `generation_rules.yaml`)
- Fuzzy deduplication (SequenceMatcher ratio ≥ 0.85 → reject)
- Exact-hash deduplication

Comments are checked for:
- Toxicity estimate within expected range per severity level
- `contains_explicit_violence` flag consistency
- Fuzzy and exact deduplication

### C. LLM severity judge

A second independent LLM call classifies each comment as 1/2/3 on the severity
scale. If the judge's label disagrees with the intended severity, the item is
flagged and (optionally) rejected.

Agreement rate is logged. If below `min_judge_agreement` (default 0.80),
a warning is raised.

### D. Realism check

A separate LLM call assesses whether each post caption sounds like authentic
Instagram content. Posts flagged as unrealistic are logged (but not automatically
rejected — this is left to researcher discretion).

---

## Configuration reference

### `configs/study_config.yaml`

Controls design factors, generation parameters (temperature, concurrency,
word counts), and validation thresholds.

### `configs/prompts.yaml`

All LLM prompts. Each entry has `system` and `user` fields. User prompts use
Python `str.format()` placeholders. Prompts are version-controlled separately
from code.

### `configs/generation_rules.yaml`

Forbidden patterns, toxicity ranges per severity level, and content rules
enforced by the schema validator.

---

## Research ethics considerations

1. **No real users**: all profiles, posts, and comments are entirely synthetic
   and fictional. No real person's data is used or represented.

2. **Original posts are hate-speech-free**: the post that survey participants
   see as the "original content" never contains hate speech, slurs, or calls
   to violence. Only the comment (the experimental manipulation) varies in
   hostility.

3. **Participant protection**: researchers using this pipeline in surveys should
   follow standard IRB protocols for exposure to harmful content, including
   content warnings, opt-out options, and debriefing.

4. **Data security**: generated stimuli contain synthetic hate speech and should
   be stored and shared according to institutional data governance policies.

5. **Misuse prevention**: this codebase is designed for academic research. The
   generated content should not be used outside the study context or published
   in ways that could be misappropriated.

---

## Running the existing CSV-based pipeline

```bash
make pipeline           # English stimuli (HTML + PNG)
make prepare            # Annotate CSVs with ideology tags
make gen-conservative   # Generate conservative profiles
make validate           # Validate profile–comment matching
make verify             # Check factorial balance
make visualise          # Input data dashboard
make inspect            # Respondent inspector
```

---

## Development

```bash
# Run a quick smoke test (dry-run, no API calls)
python src/main.py --n_profiles 6 --dry-run

# Check imports
python -c "from src.pipelines.generation_pipeline import run_pipeline; print('OK')"
```
