#!/usr/bin/env bash
# Fully-ONLINE (no precomputed queue) 30-min comparison of the three
# no-rearrangement methods: M2M (py_lns), LNS-PBS (c_lns), HBH+MLA*.
# Hypothesis under test: M2M underperforms LNS-PBS *because* of the offline
# queue; with online task generation M2M should regain its expected edge.
# Config matches the 60-min queue comparison EXCEPT: time-horizon 1800,
# inventory initialized procedurally at 30% (no --initial-inventory-file),
# and NO --use-precomputed-queue. Deadlines off; no rearrangement.
# Runs fast methods first (HBH -> M2M -> LNS-PBS), then a 3-way figure.
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

auto() { echo "data/raw_data/1800_feedback_control_30.0_fast_greedy_${1}_pbs_study_small_restricted_40_120_1.0_0.0_0.0_none_none_900.json"; }
HBH_CMP="data/raw_data/cmp_hbh_online_40bot_30pct_30min.json"
M2M_CMP="data/raw_data/cmp_m2m_online_40bot_30pct_30min.json"
LNS_CMP="data/raw_data/cmp_lnspbs_online_40bot_30pct_30min.json"

echo "=== ONLINE 3a: HBH+MLA* (no queue, 30min) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON \
  --improvement-task-assign-strategy hbh_mla_star
cp "$(auto hbh_mla_star)" "$HBH_CMP" && echo "=== ONLINE 3a DONE: $HBH_CMP ==="

echo "=== ONLINE 3b: M2M (py_lns, no queue, 30min) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON \
  --improvement-task-assign-strategy py_lns
cp "$(auto py_lns)" "$M2M_CMP" && echo "=== ONLINE 3b DONE: $M2M_CMP ==="

echo "=== ONLINE 3c: LNS-PBS (c_lns, no queue, 30min) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON \
  --improvement-task-assign-strategy c_lns
cp "$(auto c_lns)" "$LNS_CMP" && echo "=== ONLINE 3c DONE: $LNS_CMP ==="

echo "=== ONLINE 3d: 3-way throughput figure ==="
.venv/bin/python GT_grid_world/scripts/plot_throughput_3way.py \
  "$M2M_CMP" "$LNS_CMP" "$HBH_CMP" \
  data/figures/throughput_rolling_3way-online.png
echo "=== ONLINE 3-WAY FINISHED exit: $? ==="
