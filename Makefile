PY      := .venv/bin/python3
PIP     := .venv/bin/pip
COUNTRY ?= en
OUTDIR  ?= outputs/run_001
JSONL	?= $(OUTDIR)/final/stimuli.jsonl

# ── Setup ──────────────────────────────────────────────────────────────────────

.PHONY: install
install:                          ## Create venv, install deps, install Chromium
	python3 -m venv .venv
	$(PIP) install -r requirements.txt
	.venv/bin/playwright install chromium

# ── LLM generation pipeline (src/) ────────────────────────────────────────────

.PHONY: generate
generate:                         ## Run LLM generation pipeline (default: 50 profiles, seed 42)
	$(PY) src/main.py --n_profiles 50 --seed 42 --output_dir $(OUTDIR)

.PHONY: generate-dry
generate-dry:                     ## Dry-run: print design matrix, no API calls
	$(PY) src/main.py --n_profiles 50 --dry-run

.PHONY: generate-fast
generate-fast:                    ## Generate without validation judges (faster)
	$(PY) src/main.py --n_profiles 50 --seed 42 --no-judge --no-realism --output_dir outputs/run_fast

.PHONY: generate-html
generate-html:                    ## Generate + render post HTML stimuli + profile pages
	$(PY) src/main.py --n_profiles 50 --seed 42 --html --output_dir $(OUTDIR)

.PHONY: generate-screenshots
generate-screenshots:             ## Generate + render HTML + PNG screenshots (requires playwright)
	$(PY) src/main.py --n_profiles 50 --seed 42 --screenshots --output_dir $(OUTDIR)

.PHONY: render
render:                           ## Render post HTML from existing stimuli.jsonl  (JSONL=…/stimuli.jsonl)
	$(PY) src/main.py --from-jsonl $(JSONL) --seed 42

.PHONY: render-screenshots
render-screenshots:               ## Render post HTML + PNGs from existing stimuli.jsonl
	$(PY) src/main.py --from-jsonl $(JSONL) --seed 42 --screenshots

# ── English stimulus generation ────────────────────────────────────────────────

.PHONY: pipeline
pipeline:                         ## Generate HTML + PNG stimuli (English)
	$(PY) scripts/pipeline.py

.PHONY: pipeline-html
pipeline-html:                    ## Generate HTML stimuli only, skip screenshots (English)
	$(PY) scripts/pipeline.py --skip-screenshots

# ── Country adaptation ─────────────────────────────────────────────────────────
# Usage:  make adapt COUNTRY=it
#         make stimuli COUNTRY=de
#         make country COUNTRY=fr   (adapt + stimuli in one step)

.PHONY: adapt
adapt:                            ## Translate profiles + comments via Claude API  (COUNTRY=it|es|fr|de)
	$(PY) scripts/generate_country_data.py --country $(COUNTRY)

.PHONY: stimuli
stimuli:                          ## Generate HTML + PNG stimuli for a country  (COUNTRY=it|es|fr|de)
	$(PY) scripts/pipeline.py --country $(COUNTRY)

.PHONY: stimuli-html
stimuli-html:                     ## Generate HTML stimuli only for a country  (COUNTRY=it|es|fr|de)
	$(PY) scripts/pipeline.py --country $(COUNTRY) --skip-screenshots

.PHONY: country
country: adapt stimuli-html       ## Full country run: adapt content then generate HTML  (COUNTRY=it|es|fr|de)

# ── Profile–comment matching ───────────────────────────────────────────────────

.PHONY: prepare
prepare:                          ## Annotate CSVs with ideology + identity_tags + target_tag
	$(PY) scripts/prepare_data.py

.PHONY: gen-conservative
gen-conservative:                 ## Generate conservative profiles via Claude API → data/conservative_profiles.csv
	$(PY) scripts/generate_conservative_profiles.py

.PHONY: merge-conservative
merge-conservative:               ## Generate + merge conservative profiles into data/profiles.csv
	$(PY) scripts/generate_conservative_profiles.py --merge

.PHONY: validate
validate:                         ## Validate profile–comment coherence (rule-based + LLM sample)
	$(PY) scripts/validate_matching.py

.PHONY: validate-fast
validate-fast:                    ## Validate profile–comment coherence (rule-based only, no LLM)
	$(PY) scripts/validate_matching.py --no-llm

# ── Profile pages ─────────────────────────────────────────────────────────────

.PHONY: profiles-html
profiles-html:                    ## Generate profiles + render profile HTML pages
	$(PY) src/main.py --n_profiles 50 --seed 42 --html --no-judge --no-realism --output_dir $(OUTDIR)

.PHONY: profiles-screenshots
profiles-screenshots:             ## Generate profiles + render profile HTML + PNG screenshots
	$(PY) src/main.py --n_profiles 50 --seed 42 --screenshots --no-judge --no-realism --output_dir $(OUTDIR)

# ── Inspection tools ───────────────────────────────────────────────────────────

.PHONY: visualise
visualise:                        ## Build input-data dashboard  → output/visualise_inputs.html
	$(PY) scripts/visualise_inputs.py

.PHONY: inspect
inspect:                          ## Build respondent inspector  → output/inspect_respondents.html
	$(PY) scripts/inspect_respondents.py

.PHONY: verify
verify:                           ## Check factorial balance of the design
	$(PY) scripts/verify_balance.py

view-posts:                       ## Extract post topics + captions to CSV, open in Excel	
	jq  -r '.[] | [.topic, .caption] | @csv' $(OUTDIR)/raw/posts.json | sort > _posts.csv && open _posts.csv

view-comments:                       ## Extract post topics + captions to CSV, open in Excel	
	jq  -r '.[] | [.target_group, .severity, .toxicity_estimate, .text] | @csv' $(OUTDIR)/raw/comments.json | sort > _comments.csv && open _comments.csv


# ── Policies ───────────────────────────────────────────────────────────────


policies: 
	$(PY) src/main.py \
	--policies configs/policies.yaml \
	--n_profiles 28 --seed 42 \
	--output_dir outputs/run_006 \
	--model gpt-4o


# ── Housekeeping ───────────────────────────────────────────────────────────────

.PHONY: clean
clean:                            ## Remove generated output (HTML, PNG, metadata)
	rm -rf output/html output/png output/metadata output/validation
	rm -rf output/it output/es output/fr output/de

.PHONY: help
help:                             ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*##"}; {printf "  %-18s %s\n", $$1, $$2}'

.DEFAULT_GOAL := help



test2:
	export OPENAI_API_KEY=$$JRC_OPENAI_API_KEY && \
	export OPENAI_BASE_URL=https://api-gpt.jrc.ec.europa.eu/v1 && \
	.venv/bin/python src/main.py \
	--from-csv stim_df_italy.csv \
	--output_dir outputs/test_italy_v2 \
	--no-judge --no-realism \
	--model gpt-oss-120b