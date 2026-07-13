#!/usr/bin/env bash
# 1-hour, queue-based 4-way comparison (M2M, crM2M, LNS-PBS, HBH+MLA*) with the
# new physics features enabled UNIFORMLY on every method:
#   --pick-place-time --pick-place-duration 4   (4-tick pick + 4-tick place)
#   --buffer-capacity-k 20 --buffer-consumption-rate 70.0  (outbound-only buffer)
# NOTE: this is the OLD buffer 4-way (mu=70, pre buffer-aware slack). The frozen
# cmp_*_buffer_* JSONs + throughput_rolling_4way-buffer-pickplace.png are the
# reference for that run. The new buffer-aware config (mu=25, alpha read from the
# timestamped queue) lives in run_4way_buffer_aware.sh with separate output names.
# Config otherwise matches the prior no-buffer queue 4-way (3600 ticks, 40 bots,
# 30% inventory, deadlines off, seed 900, oscillating_uniform_inventory_0 queue).
# No allocator logic changes -- the features live in the shared execution layer,
# so each baseline keeps its own core behaviour.
#
# Order: M2M (py_lns) -> crM2M (py_lns) -> HBH (slow) -> LNS-PBS. M2M and crM2M
# share the py_lns auto-filename, so each is backed up immediately before the
# next run overwrites it. New cmp_*_buffer_* filenames + a new figure leave the
# no-buffer results untouched.
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
--buffer-capacity-k 20 --buffer-consumption-rate 70.0"

PYLNS_AUTO="data/raw_data/3600_feedback_control_30.0_fast_greedy_py_lns_pbs_study_small_restricted_40_120_1.0_0.0_0.0_none_none_900.json"
HBH_AUTO="data/raw_data/3600_feedback_control_30.0_fast_greedy_hbh_mla_star_pbs_study_small_restricted_40_120_1.0_0.0_0.0_none_none_900.json"
LNS_AUTO="data/raw_data/3600_feedback_control_30.0_fast_greedy_c_lns_pbs_study_small_restricted_40_120_1.0_0.0_0.0_none_none_900.json"

M2M_CMP="data/raw_data/cmp_m2m_buffer_40bot_30pct_60min.json"
CRM2M_CMP="data/raw_data/cmp_crm2m_buffer_40bot_30pct_60min.json"
HBH_CMP="data/raw_data/cmp_hbh_buffer_40bot_30pct_60min.json"
LNS_CMP="data/raw_data/cmp_lnspbs_buffer_40bot_30pct_60min.json"

echo "=== BUFFER 4WAY 1: M2M (queue, buffer+pick/place) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON --improvement-task-assign-strategy py_lns
cp "$PYLNS_AUTO" "$M2M_CMP" && echo "=== 1 DONE: $M2M_CMP ==="

echo "=== BUFFER 4WAY 2: crM2M (queue, buffer+pick/place, lambda=1.5 cutoff=10) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON --improvement-task-assign-strategy py_lns \
  --enable-rearrangement --crm2m-lambda 1.5 --crm2m-detour-cutoff 10.0
cp "$PYLNS_AUTO" "$CRM2M_CMP" && echo "=== 2 DONE: $CRM2M_CMP ==="

echo "=== BUFFER 4WAY 3: HBH+MLA* (queue, buffer+pick/place) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON --improvement-task-assign-strategy hbh_mla_star
cp "$HBH_AUTO" "$HBH_CMP" && echo "=== 3 DONE: $HBH_CMP ==="

echo "=== BUFFER 4WAY 4: LNS-PBS (queue, buffer+pick/place) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON --improvement-task-assign-strategy c_lns
cp "$LNS_AUTO" "$LNS_CMP" && echo "=== 4 DONE: $LNS_CMP ==="

echo "=== BUFFER 4WAY 5: 4-way throughput figure ==="
.venv/bin/python GT_grid_world/scripts/plot_throughput_4way.py \
  "$M2M_CMP" "$CRM2M_CMP" "$LNS_CMP" "$HBH_CMP" \
  data/figures/throughput_rolling_4way-buffer-pickplace.png

echo "=== BUFFER 4WAY THROUGHPUT SUMMARY ==="
.venv/bin/python - "$M2M_CMP" "$CRM2M_CMP" "$LNS_CMP" "$HBH_CMP" <<'PY'
import json, sys
names = ["M2M", "crM2M", "LNS-PBS", "HBH+MLA*"]
for name, fp in zip(names, sys.argv[1:]):
    d = json.load(open(fp)); t = int(d.get("timesteps_completed") or 0)
    r = int(d.get("total_completed_tasks") or 0)
    rate = r / (t / 60.0) if t else 0.0
    print(f"{name:10} {r:>6} real  {rate:>7.2f} tasks/min")
PY
echo "=== BUFFER 4WAY FINISHED exit: $? ==="
