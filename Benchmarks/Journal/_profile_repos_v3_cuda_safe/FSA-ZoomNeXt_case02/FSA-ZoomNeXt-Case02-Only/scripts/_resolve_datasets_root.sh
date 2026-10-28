#!/usr/bin/env bash
# Resolve DATASETS_ROOT for RunPod/Linux runs.
# This file is sourced by train/debug/eval scripts.

fsa_dataset_root_score() {
  local root="$1"
  local score=0
  [[ -d "${root}/COD10K-v3/Train/Image" ]] && score=$((score + 1))
  [[ -d "${root}/COD10K-v3/Train/GT_Object" ]] && score=$((score + 1))
  [[ -d "${root}/COD10K-v3/Test/Image" ]] && score=$((score + 1))
  [[ -d "${root}/COD10K-v3/Test/GT_Object" ]] && score=$((score + 1))
  [[ -d "${root}/CAMO-V.1.0-CVIU2019/Images/Train" ]] && score=$((score + 1))
  [[ -d "${root}/CAMO-V.1.0-CVIU2019/Images/Test" ]] && score=$((score + 1))
  [[ -d "${root}/CAMO-V.1.0-CVIU2019/GT" ]] && score=$((score + 1))
  [[ -d "${root}/CHAMELEON/Imgs" ]] && score=$((score + 1))
  [[ -d "${root}/CHAMELEON/GT" ]] && score=$((score + 1))
  [[ -d "${root}/NC4K/Imgs" ]] && score=$((score + 1))
  [[ -d "${root}/NC4K/GT" ]] && score=$((score + 1))
  echo "${score}"
}

fsa_resolve_datasets_root() {
  local preferred="${DATASETS_ROOT:-}"
  local candidates=()
  [[ -n "${preferred}" ]] && candidates+=("${preferred}")
  candidates+=(
    "/workspace/datasets"
    "/workspace/COD/datasets"
    "/workspace/cod/datasets"
    "/workspace/data/datasets"
    "/workspace/data"
    "$(pwd)/datasets"
    "$(dirname "$(pwd)")/datasets"
  )

  if [[ -d /workspace ]]; then
    while IFS= read -r cod_dir; do
      [[ -n "${cod_dir}" ]] && candidates+=("$(dirname "${cod_dir}")")
    done < <(find /workspace -maxdepth 6 -type d -name COD10K-v3 2>/dev/null | head -n 20)
  fi

  local best="${preferred:-/workspace/COD/datasets}"
  local best_score=-1
  local cand score
  for cand in "${candidates[@]}"; do
    [[ -z "${cand}" ]] && continue
    score="$(fsa_dataset_root_score "${cand}")"
    if (( score > best_score )); then
      best="${cand}"
      best_score="${score}"
    fi
  done

  export DATASETS_ROOT="${best}"
  echo "[FSA-ZoomNeXt] DATASETS_ROOT=${DATASETS_ROOT}  score=${best_score}/11"

  if [[ ! -d "${DATASETS_ROOT}/COD10K-v3/Train/Image" ]]; then
    echo "[FSA-ZoomNeXt][ERROR] Cannot find: ${DATASETS_ROOT}/COD10K-v3/Train/Image" >&2
    echo "[FSA-ZoomNeXt] Try:" >&2
    echo "  find /workspace -maxdepth 6 -type d -name COD10K-v3" >&2
    echo "  DATASETS_ROOT=/actual/parent/of/COD10K-v3 bash scripts/train_fsa_pvtb2.sh" >&2
    echo "[FSA-ZoomNeXt] Candidate COD10K-v3 folders under /workspace:" >&2
    find /workspace -maxdepth 6 -type d -name COD10K-v3 2>/dev/null | sed 's/^/  /' >&2 || true
    return 2
  fi
}
