#!/usr/bin/env bash
# Explicit user execution only; each stage creates a separate short plan.
set -euo pipefail
STUDY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PY=${PY:-/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python}
EXISTING="/home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study"
MANIFEST=${MANIFEST:-"$EXISTING/results/priority8-full4040-fixed-20261101_200834/manifest.json"}
PREPARE_ONLY=0
if [[ ${1:-} == --prepare-only ]]; then PREPARE_ONLY=1; shift; fi
if [[ $# -lt 2 || $# -gt 3 ]]; then
    printf '%s\n' 'Usage: bash launch_scale_mechanism.sh [--prepare-only] core|resolution|fusion|distill /absolute/weights.pth [/absolute/teacher-best.pth]' >&2
    exit 2
fi
STAGE=$1
WEIGHTS=$2
TEACHER=${3:-}
EXTRA=()
case "$STAGE" in
    core) PHASE=mechanism4; STRIDE=32 ;;
    resolution) PHASE=mechanism4; STRIDE=16 ;;
    fusion) PHASE=fusion2; STRIDE=${OUTPUT_STRIDE:-32} ;;
    distill)
        PHASE=distill2; STRIDE=${OUTPUT_STRIDE:-32}
        if [[ $TEACHER != /* || ! -s $TEACHER ]]; then
            printf '%s\n' 'distill needs the completed image_pyramid__mean best.pth as its third argument.' >&2
            exit 2
        fi
        EXTRA=(--distill-teacher "$TEACHER" --distill-weight 0.1) ;;
    *) printf 'Unknown stage: %s\n' "$STAGE" >&2; exit 2 ;;
esac
if [[ $STAGE != distill && -n $TEACHER ]]; then
    printf '%s\n' 'Only the distill stage accepts a teacher checkpoint.' >&2; exit 2
fi
for INPUT in "$PY" "$MANIFEST" "$WEIGHTS"; do
    if [[ $INPUT != /* || ! -s $INPUT ]]; then
        printf 'Required absolute input file missing: %s\n' "$INPUT" >&2; exit 2
    fi
done
STAMP=$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)
OUTPUT=${OUTPUT:-"$EXISTING/results/scale-mechanism-$STAGE-os$STRIDE-seed3407-$STAMP"}
if [[ $OUTPUT != /* || -e $OUTPUT ]]; then
    printf 'OUTPUT must be an absolute new directory: %s\n' "$OUTPUT" >&2; exit 2
fi
"$PY" "$STUDY/plan_study.py" \
    --phase "$PHASE" --seeds 3407 --backbone convnext_tiny --feature-mode last \
    --output-stride "$STRIDE" --training-protocol full_train --manifest "$MANIFEST" \
    --backbone-weights "$WEIGHTS" --python "$PY" --device "${DEVICE:-cuda:0}" \
    --size 384 --channels 48 --epochs 150 --batch-size 4 --workers 2 \
    --test-geometry resize-prediction-to-gt "${EXTRA[@]}" \
    --output "$OUTPUT/plans/$STAGE" --results "$OUTPUT/runs/$STAGE"
printf 'Result root: %s\n' "$OUTPUT"
printf 'Leaderboard: %s\n' "$OUTPUT/plans/$STAGE/leaderboard/leaderboard.html"
if (( PREPARE_ONLY )); then
    printf 'Commands only; start with: bash %q\n' "$OUTPUT/plans/$STAGE/run_all.sh"
    exit 0
fi
PYTHONUNBUFFERED=1 bash "$OUTPUT/plans/$STAGE/run_all.sh" 2>&1 | tee "$OUTPUT/run.log"
