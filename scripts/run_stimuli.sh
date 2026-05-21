#!/usr/bin/env bash
# Run the stimulus generation pipeline for a single country CSV.
#
# Usage:
#   ./scripts/run_stimuli.sh <path-to-csv> [--country COUNTRY] [extra args...]
#
# Examples:
#   ./scripts/run_stimuli.sh data/stim_df_italy.csv
#   ./scripts/run_stimuli.sh data/stim_df_italy.csv --country Italy --html --screenshots
#   ./scripts/run_stimuli.sh data/stim_df_italy.csv --no-judge --no-realism
#
# The output directory is auto-named outputs/<country>_<date>/
# Pass --output_dir PATH to override.

set -euo pipefail

CSV="${1:-}"
if [[ -z "$CSV" ]]; then
    echo "Usage: $0 <path-to-csv> [--country COUNTRY] [extra args...]"
    exit 1
fi

shift   # remaining args passed through to main.py

# Derive a default output directory from the CSV stem and today's date
STEM=$(basename "$CSV" .csv)
DATE=$(date +%Y%m%d)
DEFAULT_OUT="outputs/${STEM}_${DATE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

python src/main.py \
    --from-csv "$CSV" \
    --output_dir "$DEFAULT_OUT" \
    "$@"
