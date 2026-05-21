#!/usr/bin/env bash
# Run the stimulus generation pipeline for every CSV in a folder.
#
# Usage:
#   ./scripts/run_all_countries.sh <data-folder> [extra args...]
#
# Examples:
#   ./scripts/run_all_countries.sh data/
#   ./scripts/run_all_countries.sh data/ --html --screenshots
#   ./scripts/run_all_countries.sh data/ --no-judge
#
# Each CSV in the folder is processed in sequence.
# Output goes to outputs/<csv_stem>_<date>/

set -euo pipefail

DATA_DIR="${1:-data}"
if [[ ! -d "$DATA_DIR" ]]; then
    echo "Error: folder '$DATA_DIR' not found."
    exit 1
fi

shift || true   # remaining args passed to each run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
CSV_FILES=("$DATA_DIR"/*.csv)
shopt -u nullglob

if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
    echo "No CSV files found in $DATA_DIR"
    exit 1
fi

echo "Found ${#CSV_FILES[@]} CSV file(s) in $DATA_DIR"

for CSV in "${CSV_FILES[@]}"; do
    echo ""
    echo "============================================================"
    echo " Processing: $CSV"
    echo "============================================================"
    bash "$SCRIPT_DIR/run_stimuli.sh" "$CSV" "$@"
done

echo ""
echo "All countries done."
