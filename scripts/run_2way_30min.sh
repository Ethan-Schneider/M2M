#!/usr/bin/env bash
# 30-minute (1800-tick) 2-way M2M vs crM2M test, mapping Ethan's run_experiments.sh
# parameters onto the symbotic_2026 branch flags as closely as possible:
#   seed 0, 40 bots, max-tasks 120, freq 0.25, ratio 1.0, 30 SKUs, inv 25.0,
#   deadlines ON (normal, offset 180, weight 0.25), W 300, B 60, lambda 1.5,
#   pick/place 4, buffer K=20 mu=25.
# Queue: Ethan's own (restricted_small_uniform_with_deadlines_20000_arrival_rate_60)
# is NOT committed, so we generated an equivalent UNIFORM-tasking queue +
# matching inventory via queue_generation/queue_generator.py (8000 tasks, 25%
# full, 30 SKUs, study_small_restricted) and stamped it R=60. Uniform tasking
# (vs our bursty oscillating queue) spreads outbound demand spatially, which is
# what keeps the mu=25 buffer regime from gridlocking the PBS path planner.
# crM2M is our concatenated rearrangement (--enable-rearrangement + --crm2m-*),
# whereas Ethan's script used insertion; the numeric lambda (1.5) matches.
#
# Output names are dedicated (cmp_*_30min.json, figures under twoway_30min/) so
# nothing else is clobbered. Only M2M + crM2M run here; the 4-way comes later.
set -uo pipefail
cd /home/dcleeman/symbotic/M2M

MAP="$PWD/data/maps/study_small_restricted"
Q="$PWD/data/queues/uniform_tasking_buffer_30sku.txt"
INV="$PWD/data/initial_inventories/uniform_tasking_buffer_30sku_init_inventory.txt"

COMMON="--seed 0 --num-robots 40 --time-horizon 1800 --max-tasks 120 \
--initial-task-assign-strategy fast_greedy --path-planning-strategy pbs \
--initial-inventory 25.0 --frequency 0.25 --inbound-outbound-ratio 1.0 \
--num-skus 30 --weight-init-method uniform --map $MAP \
--cost-calculation-method shortest_path --removal-operator shaw \
--repair-operator greedy --acceptance-function simulated_annealing \
--T-0 1.0 --alpha 0.99 --deadline-generation-method normal --deadline-offset 180 \
--deadline-weight 0.25 --base-cost-weight 1.0 --sku-distribution-weight 0.0 \
--agent-unallocated-penalty 5.0 --W 300 --B 60 \
--task-gen-strategy feedback_control --use-precomputed-queue --queue-file $Q \
--initial-inventory-file $INV \
--pick-place-time --pick-place-duration 4 \
--buffer-capacity-k 20 --buffer-consumption-rate 25.0"

# Deterministic auto-name (GT_grid_world.py line 722): fields are
# T_taskgen_inv_initassign_improve_path_map_robots_maxtasks_baseW_deadlineW_skuW_repairdet_repairfn_seed
AUTO="data/raw_data/1800_feedback_control_25.0_fast_greedy_py_lns_pbs_study_small_restricted_40_120_1.0_0.25_0.0_none_none_0.json"

M2M_CMP="data/raw_data/cmp_m2m_30min.json"
CRM2M_CMP="data/raw_data/cmp_crm2m_30min.json"
FIG_DIR="data/figures/twoway_30min"

echo "=== 30MIN 2WAY 1: M2M (py_lns, buffer mu=25 + pick/place, deadlines on) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON --improvement-task-assign-strategy py_lns
cp "$AUTO" "$M2M_CMP" && echo "=== 1 DONE: $M2M_CMP ==="

echo "=== 30MIN 2WAY 2: crM2M (py_lns + rearrangement, lambda=1.5 cutoff=10) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON --improvement-task-assign-strategy py_lns \
  --enable-rearrangement --crm2m-lambda 1.5 --crm2m-detour-cutoff 10.0
cp "$AUTO" "$CRM2M_CMP" && echo "=== 2 DONE: $CRM2M_CMP ==="

echo "=== 30MIN 2WAY 3: M2M-vs-crM2M figure set ==="
.venv/bin/python GT_grid_world/scripts/plot_crm2m_compare.py "$M2M_CMP" "$CRM2M_CMP" "$FIG_DIR"

echo "=== 30MIN 2WAY THROUGHPUT SUMMARY ==="
.venv/bin/python - "$M2M_CMP" "$CRM2M_CMP" <<'PY'
import json, sys
for name, fp in zip(["M2M", "crM2M"], sys.argv[1:]):
    d = json.load(open(fp)); t = int(d.get("timesteps_completed") or 0)
    r = int(d.get("total_completed_tasks") or 0)
    rate = r / (t / 60.0) if t else 0.0
    print(f"{name:8} {r:>6} tasks  {rate:>7.2f} tasks/min  ({t} ticks)")
PY
echo "=== 30MIN 2WAY FINISHED exit: $? ==="
