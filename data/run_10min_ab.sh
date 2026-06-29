#!/bin/bash
# 30 bots / 30% capacity / 10-minute (600-tick) A/B on a schedule produced by
# the existing schedule_generation tool (schedule_30bot_30pct_10min: 1200 tasks,
# balanced 633 in / 567 out). Replays the SAME schedule under plain M2M and
# M2M+crM2M, then regenerates the six canonical M2M vs crM2M figures.
set -uo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PY="$repo_root/.venv/bin/python"
RAW="$repo_root/data/raw_data"; LOG="$repo_root/data/logs"
cd "$repo_root"; mkdir -p "$RAW" "$LOG" "$repo_root/data/figures"
MAP="$repo_root/data/maps/study_small_restricted"
SCHED="$repo_root/data/schedules/schedule_30bot_30pct_10min.txt"
INV="$repo_root/data/initial_inventories/schedule_30bot_30pct_10min_init_inventory.txt"
BASE_JSON="$RAW/600_precomputed_schedule_30.0_fast_greedy_py_lns_pbs_study_small_restricted_30_120_1.0_0.0_0.0_none_none_900.json"

common=(
  --seed 900 --num-robots 30 --time-horizon 600 --max-tasks 120
  --initial-task-assign-strategy fast_greedy --improvement-task-assign-strategy py_lns
  --path-planning-strategy pbs
  --initial-inventory 30.0 --frequency 0.25 --inbound-outbound-ratio 1.0 --num-skus 30
  --weight-init-method uniform --map "$MAP" --cost-calculation-method shortest_path
  --removal-operator shaw --repair-operator greedy --acceptance-function simulated_annealing
  --T-0 1.0 --alpha 0.99 --deadline-generation-method normal --deadline-offset 180
  --intermediate-data-interval 4000
  --base-cost-weight 1.0 --deadline-weight 0.0 --sku-distribution-weight 0.0
  --agent-unallocated-penalty 5.0
  --solution-repair-detection-function none --solution-repair-function none
  --task-gen-strategy feedback_control
  --use-precomputed-schedule --schedule-file "$SCHED" --initial-inventory-file "$INV"
)

echo "=== RUN 1/2: regular M2M (10min) ==="
"$PY" -u GT_grid_world/GT_grid_world.py "${common[@]}" > "$LOG/ten_m2m.log" 2>&1
mv -f "$BASE_JSON" "$RAW/cmp_m2m_30bot_30pct_10min.json"
echo "RUN 1 done"

echo "=== RUN 2/2: M2M + crM2M (10min) ==="
"$PY" -u GT_grid_world/GT_grid_world.py "${common[@]}" --enable-rearrangement > "$LOG/ten_crm2m.log" 2>&1
mv -f "$BASE_JSON" "$RAW/cmp_crm2m_30bot_30pct_10min.json"
echo "RUN 2 done"

echo "=== FIGURES: six canonical M2M vs crM2M ==="
(cd "$repo_root/GT_grid_world/scripts" && "$PY" plot_crm2m_compare.py \
  "$RAW/cmp_m2m_30bot_30pct_10min.json" \
  "$RAW/cmp_crm2m_30bot_30pct_10min.json" \
  "$repo_root/data/figures") > "$LOG/ten_figures.log" 2>&1
echo "FIGURES done"
echo "TEN_MIN_AB_COMPLETE"
