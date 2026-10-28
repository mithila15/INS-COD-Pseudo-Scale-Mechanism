#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
echo "[clean] removed Python cache"
