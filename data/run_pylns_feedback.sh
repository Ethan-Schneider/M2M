#!/bin/bash
# Decisive test: py_lns improver on NATIVE feedback_control (same task source as
# the 957-task c_lns reference), 30 bots / 30% / 600 ticks. Answers whether
# py_lns can reach ~957 throughput in the high-supply setting that c_lns did.
set -uo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PY="$repo_root/.venv/bin/python"
RAW="$repo_root/data/raw_data"
cd "$repo_root"
MAP="$repo_root/data/maps/study_small_restricted"
BASE_JSON="$RAW/600_feedback_control_30.0_fast_greedy_py_lns_pbs_study_small_restricted_30_120_1.0_0.0_0.0_none_none_900.json"
"$PY" -u GT_grid_world/GT_grid_world.py \
  --seed 900 --num-robots 30 --time-horizon 600 --max-tasks 120 \
  --task-gen-strategy feedback_control \
  --initial-task-assign-strategy fast_greedy \
  --improvement-task-assign-strategy py_lns \
  --path-planning-strategy pbs \
  --initial-inventory 30.0 --frequency 0.25 --inbound-outbound-ratio 1.0 --num-skus 30 \
  --weight-init-method uniform --map "$MAP" --cost-calculation-method shortest_path \
  --removal-operator shaw --repair-operator greedy --acceptance-function simulated_annealing \
  --T-0 1.0 --alpha 0.99 --deadline-generation-method normal --deadline-offset 180 \
  --intermediate-data-interval 4000 \
  --base-cost-weight 1.0 --deadline-weight 0.0 --sku-distribution-weight 0.0 \
  --agent-unallocated-penalty 5.0 \
  --solution-repair-detection-function none --solution-repair-function none \
  > "$repo_root/data/logs/run_pylns_feedback.log" 2>&1
mv -f "$BASE_JSON" "$RAW/diag_pylns_feedback_30bot_30pct_600t_seed900.json"
echo "PYLNS_FEEDBACK_DONE"
