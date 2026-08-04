#!/bin/bash
# A/B on the DENSE schedule (~1097 tasks): regular M2M vs M2M+crM2M.
# Identical config, only --enable-rearrangement differs. Outputs share a
# base filename so each is renamed immediately after its run.
set -uo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PY="$repo_root/.venv/bin/python"
RAW="$repo_root/data/raw_data"
mkdir -p "$RAW" "$repo_root/data/buffer_data" "$repo_root/data/videos" "$repo_root/data/logs"
cd "$repo_root"

MAP="$repo_root/data/maps/study_small_restricted"
SCHED="$repo_root/data/schedules/study_small_restricted_30bot_30pct_10min_dense.txt"
INV="$repo_root/data/initial_inventories/study_small_restricted_30bot_30pct_10min_dense_init_inventory.txt"
BASE_JSON="$RAW/600_precomputed_schedule_30.0_fast_greedy_py_lns_pbs_study_small_restricted_30_120_1.0_0.0_0.0_none_none_900.json"

common_args=(
  --seed 900 --num-robots 30 --time-horizon 600 --max-tasks 120
  --task-gen-strategy feedback_control
  --initial-task-assign-strategy fast_greedy
  --improvement-task-assign-strategy py_lns
  --path-planning-strategy pbs
  --initial-inventory 30.0 --frequency 0.25 --inbound-outbound-ratio 1.0 --num-skus 30
  --weight-init-method uniform --map "$MAP" --cost-calculation-method shortest_path
  --removal-operator shaw --repair-operator greedy --acceptance-function simulated_annealing
  --T-0 1.0 --alpha 0.99 --deadline-generation-method normal --deadline-offset 180
  --intermediate-data-interval 4000
  --base-cost-weight 1.0 --deadline-weight 0.0 --sku-distribution-weight 0.0
  --agent-unallocated-penalty 5.0
  --solution-repair-detection-function none --solution-repair-function none
  --use-precomputed-schedule --schedule-file "$SCHED" --initial-inventory-file "$INV"
)

echo "=== RUN 1/2: regular M2M (dense) ==="
"$PY" -u GT_grid_world/GT_grid_world.py "${common_args[@]}" > "$repo_root/data/logs/run_m2m_dense.log" 2>&1
mv -f "$BASE_JSON" "$RAW/cmp_m2m_baseline_dense_30bot_30pct_600t_seed900.json"
echo "RUN 1 done"

echo "=== RUN 2/2: M2M + crM2M (dense) ==="
"$PY" -u GT_grid_world/GT_grid_world.py "${common_args[@]}" --enable-rearrangement > "$repo_root/data/logs/run_crm2m_dense.log" 2>&1
mv -f "$BASE_JSON" "$RAW/cmp_crm2m_dense_30bot_30pct_600t_seed900.json"
echo "RUN 2 done"
echo "ALL_DENSE_RUNS_COMPLETE"
