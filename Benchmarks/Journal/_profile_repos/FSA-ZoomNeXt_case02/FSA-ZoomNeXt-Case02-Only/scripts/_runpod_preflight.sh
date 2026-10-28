#!/usr/bin/env bash
# Common preflight used by train/debug/eval scripts.
set -euo pipefail
fsa_runpod_preflight() {
  if [[ "${RUNPOD_SKIP_PREFLIGHT:-0}" == "1" ]]; then
    echo "[preflight] skipped by RUNPOD_SKIP_PREFLIGHT=1"
    return 0
  fi
  local auto_flag="--auto-fix"
  if [[ "${RUNPOD_AUTO_FIX:-1}" != "1" ]]; then
    auto_flag=""
  fi
  python tools/runpod_preflight.py ${auto_flag} --clean-pycache
}
