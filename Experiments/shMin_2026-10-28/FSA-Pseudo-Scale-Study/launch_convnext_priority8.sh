#!/usr/bin/env bash
# User-invoked training, evaluation and profiling of exactly eight cases, one seed.
# --prepare-only writes commands without starting any experiment.
set -euo pipefail

STUDY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PY=${PY:-/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python}
MANIFEST=${MANIFEST:-"$STUDY/results/priority8-full4040-fixed-20261101_200834/manifest.json"}
PREPARE_ONLY=0
if [[ ${1:-} == --prepare-only ]]; then
    PREPARE_ONLY=1
    shift
fi
if [[ $# != 1 || $1 != /* ]]; then
    printf '%s\n' 'Usage: bash launch_convnext_priority8.sh [--prepare-only] /absolute/path/to/convnext_tiny-983f1562.pth' >&2
    exit 2
fi
WEIGHTS=$1
for INPUT in "$PY" "$MANIFEST" "$WEIGHTS"; do
    if [[ $INPUT != /* || ! -f $INPUT ]]; then
        printf 'Required absolute input file is missing: %s\n' "$INPUT" >&2
        exit 2
    fi
done
STAMP=$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)
OUTPUT=${OUTPUT:-"$STUDY/results/convnext-last-priority8-seed3407-$STAMP"}
if [[ $OUTPUT != /* || -e $OUTPUT ]]; then
    printf 'OUTPUT must be an absolute NEW result directory: %s\n' "$OUTPUT" >&2
    exit 2
fi

"$PY" "$STUDY/plan_study.py" \
    --phase priority8 --seeds 3407 --backbone convnext_tiny --feature-mode last \
    --training-protocol full_train --manifest "$MANIFEST" \
    --backbone-weights "$WEIGHTS" --python "$PY" --device cuda:0 \
    --size 384 --channels 48 --epochs 150 --batch-size 4 --workers 2 \
    --test-geometry resize-prediction-to-gt \
    --output "$OUTPUT/plans/priority8" --results "$OUTPUT/runs/priority8"

printf 'Result root: %s\n' "$OUTPUT"
printf 'Leaderboard: %s\n' "$OUTPUT/plans/priority8/leaderboard/leaderboard.html"
if (( PREPARE_ONLY )); then
    printf 'Commands only. Start explicitly with: bash %q\n' "$OUTPUT/plans/priority8/run_all.sh"
    exit 0
fi
PYTHONUNBUFFERED=1 bash "$OUTPUT/plans/priority8/run_all.sh" 2>&1 | tee "$OUTPUT/run.log"
