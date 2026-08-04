#!/bin/bash
# crM2M-ONLY 1-hour run with near-infinite lambda (only Delta=0 shuffles accepted).
# 40 bots / 30% capacity / 3600 ticks, SAME balanced schedule as the prior A/B.
# Reuses the existing M2M hour run as the overlay baseline (M2M is NOT re-run).
set -uo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PY="$repo_root/.venv/bin/python"
RAW="$repo_root/data/raw_data"; LOG="$repo_root/data/logs"
cd "$repo_root"; mkdir -p "$RAW" "$LOG" "$repo_root/data/figures/crm2m_lambda_inf"
MAP="$repo_root/data/maps/study_small_restricted"
SCHED="$repo_root/data/schedules/schedule_40bot_30pct_60min.txt"
INV="$repo_root/data/initial_inventories/schedule_40bot_30pct_60min_init_inventory.txt"
EXPECT="$RAW/3600_precomputed_schedule_30.0_fast_greedy_py_lns_pbs_study_small_restricted_40_120_1.0_0.0_0.0_none_none_900.json"
OUT="$RAW/cmp_crm2m_lambdaInf_40bot_30pct_60min.json"
M2M_BASE="$RAW/cmp_m2m_40bot_30pct_60min.json"

echo "=== crM2M (lambda=1e6, only Delta=0 shuffles), 60min, 40 bots === $(date +%H:%M:%S)"
"$PY" -u GT_grid_world/GT_grid_world.py \
  --seed 900 --num-robots 40 --time-horizon 3600 --max-tasks 120 \
  --initial-task-assign-strategy fast_greedy --improvement-task-assign-strategy py_lns \
  --path-planning-strategy pbs \
  --initial-inventory 30.0 --frequency 0.25 --inbound-outbound-ratio 1.0 --num-skus 30 \
  --weight-init-method uniform --map "$MAP" --cost-calculation-method shortest_path \
  --removal-operator shaw --repair-operator greedy --acceptance-function simulated_annealing \
  --T-0 1.0 --alpha 0.99 --deadline-generation-method normal --deadline-offset 180 \
  --intermediate-data-interval 4000 \
  --base-cost-weight 1.0 --deadline-weight 0.0 --sku-distribution-weight 0.0 \
  --agent-unallocated-penalty 5.0 \
  --solution-repair-detection-function none --solution-repair-function none \
  --task-gen-strategy feedback_control \
  --use-precomputed-schedule --schedule-file "$SCHED" --initial-inventory-file "$INV" \
  --enable-rearrangement --crm2m-lambda 1000000 \
  > "$LOG/crm2m_lambdaInf.log" 2>&1
if [[ -f "$EXPECT" ]]; then mv -f "$EXPECT" "$OUT"
else newest=$(ls -t "$RAW"/3600_precomputed_schedule_*_40_*_900.json 2>/dev/null | head -1); [[ -n "$newest" ]] && mv -f "$newest" "$OUT"; fi
echo "RUN done $(date +%H:%M:%S)"

echo "=== separate figures (crM2M lambda=inf vs M2M baseline) ==="
(cd "$repo_root/GT_grid_world/scripts" && "$PY" plot_crm2m_compare.py \
  "$M2M_BASE" "$OUT" "$repo_root/data/figures/crm2m_lambda_inf") > "$LOG/crm2m_lambdaInf_figs.log" 2>&1
echo "FIGURES done $(date +%H:%M:%S)"
echo "CRM2M_LAMBDAINF_COMPLETE"
