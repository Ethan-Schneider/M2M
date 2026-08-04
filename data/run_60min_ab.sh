#!/bin/bash
# 40 bots / 30% capacity / 1-hour (3600-tick) A/B on a balanced schedule from the
# existing schedule_generation tool (schedule_40bot_30pct_60min: 7200 tasks,
# 3632 in / 3568 out). Replays the SAME schedule under plain M2M and M2M+crM2M,
# then regenerates the six canonical M2M vs crM2M figures.
set -uo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PY="$repo_root/.venv/bin/python"
RAW="$repo_root/data/raw_data"; LOG="$repo_root/data/logs"
cd "$repo_root"; mkdir -p "$RAW" "$LOG" "$repo_root/data/figures"
MAP="$repo_root/data/maps/study_small_restricted"
SCHED="$repo_root/data/schedules/schedule_40bot_30pct_60min.txt"
INV="$repo_root/data/initial_inventories/schedule_40bot_30pct_60min_init_inventory.txt"
EXPECT="$RAW/3600_precomputed_schedule_30.0_fast_greedy_py_lns_pbs_study_small_restricted_40_120_1.0_0.0_0.0_none_none_900.json"

common=(
  --seed 900 --num-robots 40 --time-horizon 3600 --max-tasks 120
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

# mv produced json to $1; fall back to newest matching glob if exact name differs
collect() {
  if [[ -f "$EXPECT" ]]; then mv -f "$EXPECT" "$1"
  else newest=$(ls -t "$RAW"/3600_precomputed_schedule_*_40_*_900.json 2>/dev/null | head -1)
       [[ -n "$newest" ]] && mv -f "$newest" "$1"; fi
}

echo "=== RUN 1/2: regular M2M (60min, 40 bots) === $(date +%H:%M:%S)"
"$PY" -u GT_grid_world/GT_grid_world.py "${common[@]}" > "$LOG/hour_m2m.log" 2>&1
collect "$RAW/cmp_m2m_40bot_30pct_60min.json"
echo "RUN 1 done $(date +%H:%M:%S)"

echo "=== RUN 2/2: M2M + crM2M (60min, 40 bots) === $(date +%H:%M:%S)"
"$PY" -u GT_grid_world/GT_grid_world.py "${common[@]}" --enable-rearrangement > "$LOG/hour_crm2m.log" 2>&1
collect "$RAW/cmp_crm2m_40bot_30pct_60min.json"
echo "RUN 2 done $(date +%H:%M:%S)"

echo "=== FIGURES: six canonical M2M vs crM2M ==="
(cd "$repo_root/GT_grid_world/scripts" && "$PY" plot_crm2m_compare.py \
  "$RAW/cmp_m2m_40bot_30pct_60min.json" \
  "$RAW/cmp_crm2m_40bot_30pct_60min.json" \
  "$repo_root/data/figures") > "$LOG/hour_figures.log" 2>&1
echo "FIGURES done $(date +%H:%M:%S)"
echo "HOUR_AB_COMPLETE"
