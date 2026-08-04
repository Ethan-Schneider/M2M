#!/usr/bin/env bash
# NEW buffer-aware 1-hour, queue-based 4-way comparison (M2M, crM2M, LNS-PBS,
# HBH+MLA*). Same features as run_4way_buffer_pickplace.sh EXCEPT the buffer
# regime is the corrected one (per Ethan) plus the buffer-aware crM2M slack:
#   --pick-place-time --pick-place-duration 4        (4-tick pick + 4-tick place)
#   --buffer-capacity-k 20 --buffer-consumption-rate 25.0   (drain mu = 25/min)
#   queue timestamped at R=60 tasks/min -> alpha read from the queue clock
#     (estimate_arrival_rate); outbound offered ~30/min > 25/min drain, so the
#     buffer is intentionally oversubscribed (the "buffer is the bottleneck"
#     regime where buffer-aware rearrangement should pay off).
# Config otherwise matches the old buffer 4-way (3600 ticks, 40 bots, 30%
# inventory, deadlines off, seed 900, oscillating_uniform_inventory_0 queue).
#
# OUTPUTS ARE SEPARATELY NAMED (cmp_*_bufaware_*, throughput_rolling_4way-buffer-
# aware.png) so the old buffer 4-way artifacts stay untouched for comparison.
set -uo pipefail
cd /home/dcleeman/symbotic/M2M

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
--initial-inventory-file $INV \
--pick-place-time --pick-place-duration 4 \
--buffer-capacity-k 20 --buffer-consumption-rate 25.0"

PYLNS_AUTO="data/raw_data/3600_feedback_control_30.0_fast_greedy_py_lns_pbs_study_small_restricted_40_120_1.0_0.0_0.0_none_none_900.json"
HBH_AUTO="data/raw_data/3600_feedback_control_30.0_fast_greedy_hbh_mla_star_pbs_study_small_restricted_40_120_1.0_0.0_0.0_none_none_900.json"
LNS_AUTO="data/raw_data/3600_feedback_control_30.0_fast_greedy_c_lns_pbs_study_small_restricted_40_120_1.0_0.0_0.0_none_none_900.json"

M2M_CMP="data/raw_data/cmp_m2m_bufaware_40bot_30pct_60min.json"
CRM2M_CMP="data/raw_data/cmp_crm2m_bufaware_40bot_30pct_60min.json"
HBH_CMP="data/raw_data/cmp_hbh_bufaware_40bot_30pct_60min.json"
LNS_CMP="data/raw_data/cmp_lnspbs_bufaware_40bot_30pct_60min.json"

echo "=== BUFFER-AWARE 4WAY 1: M2M (queue, buffer mu=25 + pick/place) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON --improvement-task-assign-strategy py_lns
cp "$PYLNS_AUTO" "$M2M_CMP" && echo "=== 1 DONE: $M2M_CMP ==="

echo "=== BUFFER-AWARE 4WAY 2: crM2M (queue, buffer-aware slack, lambda=1.5 cutoff=10) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON --improvement-task-assign-strategy py_lns \
  --enable-rearrangement --crm2m-lambda 1.5 --crm2m-detour-cutoff 10.0
cp "$PYLNS_AUTO" "$CRM2M_CMP" && echo "=== 2 DONE: $CRM2M_CMP ==="

echo "=== BUFFER-AWARE 4WAY 3: HBH+MLA* (queue, buffer mu=25 + pick/place) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON --improvement-task-assign-strategy hbh_mla_star
cp "$HBH_AUTO" "$HBH_CMP" && echo "=== 3 DONE: $HBH_CMP ==="

echo "=== BUFFER-AWARE 4WAY 4: LNS-PBS (queue, buffer mu=25 + pick/place) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON --improvement-task-assign-strategy c_lns
cp "$LNS_AUTO" "$LNS_CMP" && echo "=== 4 DONE: $LNS_CMP ==="

echo "=== BUFFER-AWARE 4WAY 5: 4-way throughput figure ==="
.venv/bin/python GT_grid_world/scripts/plot_throughput_4way.py \
  "$M2M_CMP" "$CRM2M_CMP" "$LNS_CMP" "$HBH_CMP" \
  data/figures/throughput_rolling_4way-buffer-aware.png

echo "=== BUFFER-AWARE 4WAY THROUGHPUT SUMMARY ==="
.venv/bin/python - "$M2M_CMP" "$CRM2M_CMP" "$LNS_CMP" "$HBH_CMP" <<'PY'
import json, sys
names = ["M2M", "crM2M", "LNS-PBS", "HBH+MLA*"]
for name, fp in zip(names, sys.argv[1:]):
    d = json.load(open(fp)); t = int(d.get("timesteps_completed") or 0)
    r = int(d.get("total_completed_tasks") or 0)
    rate = r / (t / 60.0) if t else 0.0
    print(f"{name:10} {r:>6} real  {rate:>7.2f} tasks/min")
PY
echo "=== BUFFER-AWARE 4WAY FINISHED exit: $? ==="
