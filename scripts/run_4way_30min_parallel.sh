#!/usr/bin/env bash
# PARALLEL 30-minute (1800-tick) 4-way comparison: M2M, crM2M, HBH+MLA*, LNS-PBS.
#
# Identical config to run_4way_30min.sh, but the four methods run CONCURRENTLY
# (the box has 14 cores; each run is ~1 core, so there is no contention and the
# wall-clock collapses to roughly one phase instead of four back-to-back).
#
# The only wrinkle vs. the sequential script: M2M and crM2M share the py_lns
# improvement strategy and therefore the same auto-generated output name, so we
# pass an explicit --output-file to every run (added to GT_grid_world.py) so the
# four processes never clobber each other's JSON / intermediate files.
#
# Orchestration:
#   1) Launch all four runs in the background.
#   2) wait for M2M + crM2M -> emit the M2M-vs-crM2M figure set (9 shared + 2
#      crM2M-only rearrangement) so it can be reviewed while the baselines run.
#   3) wait for HBH + LNS-PBS -> emit the full 4-way figure set (shared only).
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

M2M_CMP="data/raw_data/cmp_m2m_4way30.json"
CRM2M_CMP="data/raw_data/cmp_crm2m_4way30.json"
HBH_CMP="data/raw_data/cmp_hbh_4way30.json"
LNS_CMP="data/raw_data/cmp_lnspbs_4way30.json"

TWOWAY_DIR="data/figures/twoway_30min"
FOURWAY_DIR="data/figures/fourway_30min"
LOGDIR="/tmp/run_4way_parallel"
mkdir -p "$LOGDIR"

echo "=== PARALLEL 4WAY30: launching all four runs ($(date)) ==="

.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON \
  --improvement-task-assign-strategy py_lns \
  --output-file "$M2M_CMP" > "$LOGDIR/m2m.log" 2>&1 &
M2M_PID=$!
echo "  M2M      pid $M2M_PID -> $M2M_CMP"

.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON \
  --improvement-task-assign-strategy py_lns \
  --enable-rearrangement --crm2m-lambda 1.5 --crm2m-detour-cutoff 10.0 \
  --output-file "$CRM2M_CMP" > "$LOGDIR/crm2m.log" 2>&1 &
CRM2M_PID=$!
echo "  crM2M    pid $CRM2M_PID -> $CRM2M_CMP"

.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON \
  --improvement-task-assign-strategy hbh_mla_star \
  --output-file "$HBH_CMP" > "$LOGDIR/hbh.log" 2>&1 &
HBH_PID=$!
echo "  HBH+MLA* pid $HBH_PID -> $HBH_CMP"

.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON \
  --improvement-task-assign-strategy c_lns \
  --output-file "$LNS_CMP" > "$LOGDIR/lnspbs.log" 2>&1 &
LNS_PID=$!
echo "  LNS-PBS  pid $LNS_PID -> $LNS_CMP"

# --- Phase A: as soon as M2M + crM2M finish, emit the 2-way figures. ---
wait "$M2M_PID"; M2M_RC=$?
wait "$CRM2M_PID"; CRM2M_RC=$?
echo "=== M2M (rc=$M2M_RC) + crM2M (rc=$CRM2M_RC) DONE -> 2-way figures ($(date)) ==="
if [[ $M2M_RC -eq 0 && $CRM2M_RC -eq 0 ]]; then
  .venv/bin/python GT_grid_world/scripts/plot_crm2m_compare.py "$M2M_CMP" "$CRM2M_CMP" "$TWOWAY_DIR"
  .venv/bin/python GT_grid_world/scripts/plot_crm2m_rearrangement.py "$CRM2M_CMP" "$TWOWAY_DIR" --tag crm2m
  echo "=== 2-WAY FIGURES READY in $TWOWAY_DIR ==="
else
  echo "!!! M2M or crM2M failed; skipping 2-way figures. Check $LOGDIR."
fi

# --- Phase B: wait for the two baselines, then emit the full 4-way figures. ---
wait "$HBH_PID"; HBH_RC=$?
wait "$LNS_PID"; LNS_RC=$?
echo "=== HBH (rc=$HBH_RC) + LNS-PBS (rc=$LNS_RC) DONE -> 4-way figures ($(date)) ==="
if [[ $HBH_RC -eq 0 && $LNS_RC -eq 0 && $M2M_RC -eq 0 && $CRM2M_RC -eq 0 ]]; then
  .venv/bin/python GT_grid_world/scripts/plot_4way_all.py \
    "$M2M_CMP" "$CRM2M_CMP" "$LNS_CMP" "$HBH_CMP" "$FOURWAY_DIR"
  echo "=== 4-WAY FIGURES READY in $FOURWAY_DIR ==="
else
  echo "!!! One or more runs failed; skipping 4-way figures. Check $LOGDIR."
fi

echo "=== PARALLEL 4WAY30 THROUGHPUT SUMMARY ==="
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
    blk = int(d.get("outbound_buffer_placement_blocks") or 0)
    print(f"{name:10} {r:>6} real  {rate:>7.2f} tasks/min  | shuffles: {sh}  | buffer-blocks: {blk}")
PY
echo "=== PARALLEL 4WAY30 FINISHED ($(date)) ==="
