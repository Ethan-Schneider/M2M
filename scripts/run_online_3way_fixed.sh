#!/usr/bin/env bash
# Re-run the two baselines (HBH+MLA*, LNS-PBS) fully ONLINE with the M2M
# base-cost fix in place (agent_task_sequence_time removed), then build a new
# 3-way throughput figure alongside the already-fixed M2M run. Config is
# identical to run_online_3way.sh (1800 ticks, 40 bots, 30% online, deadlines
# off, seed 900); only the source code differs (fix applied). HBH first (fast),
# then LNS-PBS (slow). M2M-fixed already exists (cmp_m2m_online_FIXED...).
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
HBH_CMP="data/raw_data/cmp_hbh_online_FIXED_40bot_30pct_30min.json"
LNS_CMP="data/raw_data/cmp_lnspbs_online_FIXED_40bot_30pct_30min.json"
M2M_CMP="data/raw_data/cmp_m2m_online_FIXED_40bot_30pct_30min.json"

echo "=== FIXED 3a: HBH+MLA* (online, fix) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON --improvement-task-assign-strategy hbh_mla_star
cp "$(auto hbh_mla_star)" "$HBH_CMP" && echo "=== FIXED 3a DONE: $HBH_CMP ==="

echo "=== FIXED 3b: LNS-PBS (online, fix) ==="
.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON --improvement-task-assign-strategy c_lns
cp "$(auto c_lns)" "$LNS_CMP" && echo "=== FIXED 3b DONE: $LNS_CMP ==="

echo "=== FIXED 3c: 3-way throughput figure (fixed) ==="
.venv/bin/python GT_grid_world/scripts/plot_throughput_3way.py \
  "$M2M_CMP" "$LNS_CMP" "$HBH_CMP" \
  data/figures/throughput_rolling_3way-online-FIXED.png

echo "=== FIXED 3-WAY THROUGHPUT (all three) ==="
.venv/bin/python - <<PY
import json
for name,fp in [("M2M",M2M_CMP if False else "data/raw_data/cmp_m2m_online_FIXED_40bot_30pct_30min.json"),
                ("LNS-PBS","data/raw_data/cmp_lnspbs_online_FIXED_40bot_30pct_30min.json"),
                ("HBH+MLA*","data/raw_data/cmp_hbh_online_FIXED_40bot_30pct_30min.json")]:
    d=json.load(open(fp)); t=int(d.get("timesteps_completed") or 0); r=int(d.get("total_completed_tasks") or 0)
    print(f"{name:10} {r:>6} real  {r/(t/60.0):>7.2f} tasks/min")
PY
echo "=== FIXED 3-WAY FINISHED exit: $? ==="
