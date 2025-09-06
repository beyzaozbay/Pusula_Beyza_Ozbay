#!/usr/bin/env bash
# Run the full pipeline end-to-end
set -euo pipefail

echo "▶ 0) Ensure output folders exist"
mkdir -p reports/summary

echo "▶ 1) Load & spec check"
python -m src.data.load_and_check

echo "▶ 2) Derive numeric columns"
python -m src.features.derive_numeric

echo "▶ 3) EDA figures & summaries"
python -m src.visualization.eda_report

echo "▶ 4) Preprocessing → model-ready dataset"
python -m src.features.preprocess

echo "✅ All steps completed."
