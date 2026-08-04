#!/usr/bin/env bash
# 30-minute (1800-tick) 4-way comparison: M2M, crM2M, HBH+MLA*, LNS-PBS.
#
# Params (per request): 30% initial inventory, 40 bots, R=60 tasks/min (queue
# arrival clock -> alpha), K=20 buffer, mu=25 drain, pick/place 4 (c_pp=8),
# deadlines ON (normal, offset 180, weight 0.25), lambda 1.5, cutoff 10, seed 0.
#
# Queue: uniform-tasking pair generated at 30% full (queue_generator.py) and
# stamped R=60 (add_queue_timestamps.py). Uniform tasking keeps the mu=25 buffer
# regime from gridlocking the PBS planner.
#
# Orchestration:
#   1) M2M + crM2M run first.
#   2) As soon as both finish -> the M2M-vs-crM2M figure set is generated
#      (6 shared metrics + 2 crM2M-only rearrangement figures) so it can be
#      reviewed while the baselines run.
#   3) HBH+MLA* + LNS-PBS run.
#   4) Full 4-way figure set (shared metrics only; rearrangement figures are
#      crM2M-specific and excluded for the non-rearranging baselines).
#
# Outputs are dedicated (cmp_*_4way30.json; figures under twoway_30min/ and
# fourway_30min/) so nothing else is clobbered.
set -uo pipefail
cd /home/dcleeman/symbotic/M2M

MAP="$PWD/data/maps/study_small_restricted"
Q="$PWD/data/queues/uniform_tasking_buffer_30sku_inv30.txt"
INV="$PWD/data/initial_inventories/uniform_tasking_buffer_30sku_inv30_init_inventory.txt"

COMMON="--seed 0 --num-robots 40 --time-horizon 1800 --max-tasks 120 \
--initial-task-assign-strategy fast_greedy --path-planning-strategy pbs \
--initial-inventory 30.0 --frequency 0.25 --inbound-outbound-ratio 1.0 \
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

# Deterministic auto-names (GT_grid_world.py): fields are
# T_taskgen_inv_initassign_improve_path_map_robots_maxtasks_ratio_deadlineW_skuW_repairdet_repairfn_seed
PYLNS_AUTO="data/raw_data/1800_feedback_control_30.0_fast_greedy_py_lns_pbs_study_small_restricted_40_120_1.0_0.25_0.0_none_none_0.json"
HBH_AUTO="data/raw_data/1800_feedback_control_30.0_fast_greedy_hbh_mla_star_pbs_study_small_restricted_40_120_1.0_0.25_0.0_none_none_0.json"
LNS_AUTO="data/raw_data/1800_feedback_control_30.0_fast_greedy_c_lns_pbs_study_small_restricted_40_120_1.0_0.25_0.0_none_none_0.json"

M2M_CMP="data/raw_data/cmp_m2m_4way30.json"
CRM2M_CMP="data/raw_data/cmp_crm2m_4way30.json"
HBH_CMP="data/raw_data/cmp_hbh_4way30.json"
LNS_CMP="data/raw_data/cmp_lnspbs_4way30.json"

TWOWAY_DIR="data/figures/twoway_30min"
FOURWAY_DIR="data/figures/fourway_30min"

echo "=== 4WAY30 1: M2M (py_lns, buffer mu=25 + pick/place, deadlines on) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON --improvement-task-assign-strategy py_lns
cp "$PYLNS_AUTO" "$M2M_CMP" && echo "=== 1 DONE: $M2M_CMP ==="

echo "=== 4WAY30 2: crM2M (py_lns + rearrangement, lambda=1.5 cutoff=10) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON --improvement-task-assign-strategy py_lns \
  --enable-rearrangement --crm2m-lambda 1.5 --crm2m-detour-cutoff 10.0
cp "$PYLNS_AUTO" "$CRM2M_CMP" && echo "=== 2 DONE: $CRM2M_CMP ==="

echo "=== 4WAY30 3: M2M-vs-crM2M figure set (6 shared + 2 crM2M rearrangement) ==="
.venv/bin/python GT_grid_world/scripts/plot_crm2m_compare.py "$M2M_CMP" "$CRM2M_CMP" "$TWOWAY_DIR"
.venv/bin/python GT_grid_world/scripts/plot_crm2m_rearrangement.py "$CRM2M_CMP" "$TWOWAY_DIR" --tag crm2m
echo "=== 2-WAY FIGURES READY in $TWOWAY_DIR ==="

echo "=== 4WAY30 4: HBH+MLA* (buffer mu=25 + pick/place) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON --improvement-task-assign-strategy hbh_mla_star
cp "$HBH_AUTO" "$HBH_CMP" && echo "=== 4 DONE: $HBH_CMP ==="

echo "=== 4WAY30 5: LNS-PBS (buffer mu=25 + pick/place) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON --improvement-task-assign-strategy c_lns
cp "$LNS_AUTO" "$LNS_CMP" && echo "=== 5 DONE: $LNS_CMP ==="

echo "=== 4WAY30 6: full 4-way figure set (shared metrics only) ==="
.venv/bin/python GT_grid_world/scripts/plot_4way_all.py \
  "$M2M_CMP" "$CRM2M_CMP" "$LNS_CMP" "$HBH_CMP" "$FOURWAY_DIR"

echo "=== 4WAY30 THROUGHPUT SUMMARY ==="
.venv/bin/python - "$M2M_CMP" "$CRM2M_CMP" "$LNS_CMP" "$HBH_CMP" <<'PY'
import json, sys
names = ["M2M", "crM2M", "LNS-PBS", "HBH+MLA*"]
for name, fp in zip(names, sys.argv[1:]):
    try:
        d = json.load(open(fp))
    except FileNotFoundError:
        print(f"{name:10} (missing {fp})"); continue
    t = int(d.get("timesteps_completed") or 0)
    r = int(d.get("total_completed_tasks") or 0)
    rate = r / (t / 60.0) if t else 0.0
    sh = int(d.get("total_completed_rearrangement_tasks") or 0)
    print(f"{name:10} {r:>6} real  {rate:>7.2f} tasks/min  | shuffles completed: {sh}")
PY
echo "=== 4WAY30 FINISHED exit: $? ==="
