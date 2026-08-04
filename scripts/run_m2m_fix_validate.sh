#!/usr/bin/env bash
set -uo pipefail
cd /home/dcleeman/symbotic/M2M
MAP="$PWD/data/maps/study_small_restricted"
COMMON="--seed 900 --num-robots 40 --time-horizon 1800 --max-tasks 120 \
--initial-task-assign-strategy fast_greedy --path-planning-strategy pbs \
--initial-inventory 30.0 --frequency 0.25 --inbound-outbound-ratio 1.0 \
--num-skus 30 --weight-init-method uniform --map $MAP \
--cost-calculation-method shortest_path --removal-operator shaw \
--repair-operator greedy --acceptance-function simulated_annealing \
--T-0 1.0 --alpha 0.99 --deadline-generation-method none --deadline-weight 0.0 \
--base-cost-weight 1.0 --sku-distribution-weight 0.0 --agent-unallocated-penalty 5.0 \
--task-gen-strategy feedback_control"
AUTO="data/raw_data/1800_feedback_control_30.0_fast_greedy_py_lns_pbs_study_small_restricted_40_120_1.0_0.0_0.0_none_none_900.json"
FIXED="data/raw_data/cmp_m2m_online_FIXED_40bot_30pct_30min.json"
echo "=== M2M FIX VALIDATION: online py_lns, seq-time term removed ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON --improvement-task-assign-strategy py_lns
cp "$AUTO" "$FIXED"
echo "=== DONE: $FIXED ==="
.venv/bin/python - <<PY
import json
d=json.load(open("$FIXED"))
ticks=int(d.get("timesteps_completed") or 0); real=int(d.get("total_completed_tasks") or 0)
print(f"M2M online (FIXED): {ticks} ticks, {real} real tasks, {real/(ticks/60.0):.2f} tasks/min")
print("pre-fix was 114.53 tasks/min; expected ~126-127")
PY
echo "=== M2M FIX VALIDATION FINISHED exit: $? ==="
