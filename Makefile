PY      := .venv/bin/python3.13
PIP     := .venv/bin/pip
COUNTRY ?= en

# ── Setup ──────────────────────────────────────────────────────────────────────

.PHONY: install
install:                          ## Create venv, install deps, install Chromium
	python3.13 -m venv .venv
	$(PIP) install numpy pandas jinja2 playwright anthropic
	.venv/bin/playwright install chromium

# ── English stimulus generation ────────────────────────────────────────────────

.PHONY: pipeline
pipeline:                         ## Generate HTML + PNG stimuli (English)
	$(PY) pipeline.py

.PHONY: pipeline-html
pipeline-html:                    ## Generate HTML stimuli only, skip screenshots (English)
	$(PY) pipeline.py --skip-screenshots

# ── Country adaptation ─────────────────────────────────────────────────────────
# Usage:  make adapt COUNTRY=it
#         make stimuli COUNTRY=de
#         make country COUNTRY=fr   (adapt + stimuli in one step)

.PHONY: adapt
adapt:                            ## Translate profiles + comments via Claude API  (COUNTRY=it|es|fr|de)
	$(PY) generate_country_data.py --country $(COUNTRY)

.PHONY: stimuli
stimuli:                          ## Generate HTML + PNG stimuli for a country  (COUNTRY=it|es|fr|de)
	$(PY) pipeline.py --country $(COUNTRY)

.PHONY: stimuli-html
stimuli-html:                     ## Generate HTML stimuli only for a country  (COUNTRY=it|es|fr|de)
	$(PY) pipeline.py --country $(COUNTRY) --skip-screenshots

.PHONY: country
country: adapt stimuli-html       ## Full country run: adapt content then generate HTML  (COUNTRY=it|es|fr|de)

# ── Profile–comment matching ───────────────────────────────────────────────────

.PHONY: prepare
prepare:                          ## Annotate CSVs with ideology + identity_tags + target_tag
	$(PY) prepare_data.py

.PHONY: gen-conservative
gen-conservative:                 ## Generate conservative profiles via Claude API → data/conservative_profiles.csv
	$(PY) generate_conservative_profiles.py

.PHONY: merge-conservative
merge-conservative:               ## Generate + merge conservative profiles into data/profiles.csv
	$(PY) generate_conservative_profiles.py --merge

.PHONY: validate
validate:                         ## Validate profile–comment coherence (rule-based + LLM sample)
	$(PY) validate_matching.py

.PHONY: validate-fast
validate-fast:                    ## Validate profile–comment coherence (rule-based only, no LLM)
	$(PY) validate_matching.py --no-llm

# ── Inspection tools ───────────────────────────────────────────────────────────

.PHONY: visualise
visualise:                        ## Build input-data dashboard  → output/visualise_inputs.html
	$(PY) visualise_inputs.py

.PHONY: inspect
inspect:                          ## Build respondent inspector  → output/inspect_respondents.html
	$(PY) inspect_respondents.py

.PHONY: verify
verify:                           ## Check factorial balance of the design
	$(PY) verify_balance.py

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
