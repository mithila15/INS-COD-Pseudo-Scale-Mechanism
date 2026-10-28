#!/usr/bin/env bash
set -euo pipefail
USE_AMP=0 bash "$(dirname "$0")/train_brg_lite_pvtb2.sh" "$@"
