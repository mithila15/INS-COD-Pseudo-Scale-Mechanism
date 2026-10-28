#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs_case02}"
"${PYTHON_BIN}" tools/summarize_case02_metrics.py --output-root "${OUTPUT_ROOT}" "$@"
