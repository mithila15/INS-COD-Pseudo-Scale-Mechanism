#!/usr/bin/env bash
set -euo pipefail
# Explicit execution: starts fresh full-training runs.
/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python /home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/run_plan.py --plan /home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/priority8-full4040-fixed-20261101_200834/plans/priority8/plan.json --actions train --skip-completed
