#!/usr/bin/env bash
set -euo pipefail
/home/master/mithila/INS_COD/.venv/fsa-study-clean/bin/python /home/master/mithila/INS_COD/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/run_plan.py --plan /home/master/mithila/shMin/Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study/results/priority8-separate-20261030_204647/plans/priority8/plan.json --actions train evaluate profile --skip-completed
