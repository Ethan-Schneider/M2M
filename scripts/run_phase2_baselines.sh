#!/usr/bin/env bash
# Phase 2: after the running M2M-vs-crM2M pipeline finishes, run the two
# remaining baselines (HBH+MLA*, LNS-PBS) for 1 hour each on the queue with
# the SAME config as the comparison, then build a single 4-way throughput
# figure (M2M, crM2M, LNS-PBS, HBH+MLA*). Deadlines off; no rearrangement on
# the baselines. Does NOT touch the six *_crm2m-vs-m2m.png figures.
set -uo pipefail
cd /home/dcleeman/symbotic/M2M

PHASE1_LOG="data/logs_m2m_vs_crm2m_60min.txt"
SIM_PROC="GT_grid_world/GT_grid_world.py"

echo "=== PHASE 2 WAIT: blocking on phase-1 pipeline ==="
# The phase-1 command runs under the persistent interactive terminal shell, so
# we can't wait on a pid. Instead poll the phase-1 log for its final success
# marker ("PHASE 3 DONE"), with a failsafe: if the sim process disappears for
# two consecutive checks without that marker, phase 1 failed -> abort.
miss=0
while true; do
  if grep -q "PHASE 3 DONE" "$PHASE1_LOG" 2>/dev/null; then
    echo "phase-1 complete (PHASE 3 DONE seen)"; break
  fi
  if pgrep -f "$SIM_PROC" >/dev/null 2>&1; then
    miss=0
  else
    miss=$((miss + 1))
    if [[ $miss -ge 2 ]]; then
      echo "phase-1 sim process gone without completion marker; aborting" >&2
      exit 1
    fi
  fi
  sleep 30
done
# Give the filesystem a moment, then confirm phase-1 actually produced outputs.
sleep 5
if [[ ! -s data/raw_data/cmp_m2m_queue_40bot_30pct_60min.json || \
      ! -s data/raw_data/cmp_crm2m_queue_40bot_30pct_60min.json ]]; then
  echo "ABORT: phase-1 comparison JSONs missing; not running phase 2." >&2
  exit 1
fi
echo "=== PHASE 2 START: phase-1 outputs present ==="

MAP="$PWD/data/maps/study_small_restricted"
Q="$PWD/data/queues/oscillating_uniform_inventory_0.txt"
INV="$PWD/data/initial_inventories/oscillating_uniform_inventory_0_init_inventory.txt"

COMMON="--seed 900 --num-robots 40 --time-horizon 3600 --max-tasks 120 \
--initial-task-assign-strategy fast_greedy --path-planning-strategy pbs \
--initial-inventory 30.0 --frequency 0.25 --inbound-outbound-ratio 1.0 \
--num-skus 30 --weight-init-method uniform --map $MAP \
--cost-calculation-method shortest_path --removal-operator shaw \
--repair-operator greedy --acceptance-function simulated_annealing \
--T-0 1.0 --alpha 0.99 --deadline-generation-method none --deadline-weight 0.0 \
--base-cost-weight 1.0 --sku-distribution-weight 0.0 --agent-unallocated-penalty 5.0 \
--task-gen-strategy feedback_control --use-precomputed-queue --queue-file $Q \
--initial-inventory-file $INV"

HBH_AUTO="data/raw_data/3600_feedback_control_30.0_fast_greedy_hbh_mla_star_pbs_study_small_restricted_40_120_1.0_0.0_0.0_none_none_900.json"
LNS_AUTO="data/raw_data/3600_feedback_control_30.0_fast_greedy_c_lns_pbs_study_small_restricted_40_120_1.0_0.0_0.0_none_none_900.json"
HBH_CMP="data/raw_data/cmp_hbh_queue_40bot_30pct_60min.json"
LNS_CMP="data/raw_data/cmp_lnspbs_queue_40bot_30pct_60min.json"

echo "=== PHASE 2a: HBH+MLA* (queue, 1h) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON \
  --improvement-task-assign-strategy hbh_mla_star
cp "$HBH_AUTO" "$HBH_CMP" && echo "=== PHASE 2a DONE: $HBH_CMP ==="

echo "=== PHASE 2b: LNS-PBS (queue, 1h) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON \
  --improvement-task-assign-strategy c_lns
cp "$LNS_AUTO" "$LNS_CMP" && echo "=== PHASE 2b DONE: $LNS_CMP ==="

echo "=== PHASE 2c: 4-way throughput figure ==="
.venv/bin/python GT_grid_world/scripts/plot_throughput_4way.py \
  data/raw_data/cmp_m2m_queue_40bot_30pct_60min.json \
  data/raw_data/cmp_crm2m_queue_40bot_30pct_60min.json \
  "$LNS_CMP" "$HBH_CMP" \
  data/figures/throughput_rolling_4way-baselines.png
echo "=== PHASE 2 FINISHED exit: $? ==="
