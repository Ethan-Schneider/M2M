#!/usr/bin/env bash
# PARALLEL 30-minute (1800-tick) 3-way comparison: M2M vs crM2M vs irM2M.
#
# Post-merge sanity run: confirm all three task-allocation methods run to
# completion on the same regime and produce comparable data. crM2M uses
# return margin m = 0 (no headroom requirement) since this is a "do they all
# work / does the data look normal" check, not a tuned comparison.
#
# All three share the K=20 / mu=25 / 40-bot / 30%-inventory regime and the
# same auto-name, so each is given an explicit --output-file.
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

M2M_CMP="data/raw_data/cmp_m2m_3way30.json"
CRM2M_CMP="data/raw_data/cmp_crm2m_3way30.json"
IRM2M_CMP="data/raw_data/cmp_irm2m_3way30.json"

LOGDIR="/tmp/run_3way_30min"
mkdir -p "$LOGDIR" data/raw_data

echo "=== 3WAY 30min: launching M2M + crM2M(m=0) + irM2M ($(date)) ==="

.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON \
  --improvement-task-assign-strategy M2M \
  --output-file "$M2M_CMP" > "$LOGDIR/m2m.log" 2>&1 &
M2M_PID=$!
echo "  M2M    pid $M2M_PID -> $M2M_CMP"

.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON \
  --improvement-task-assign-strategy M2M \
  --enable-rearrangement --crm2m-lambda 1.5 --crm2m-detour-cutoff 10.0 \
  --crm2m-return-margin 0.0 \
  --output-file "$CRM2M_CMP" > "$LOGDIR/crm2m.log" 2>&1 &
CRM2M_PID=$!
echo "  crM2M  pid $CRM2M_PID -> $CRM2M_CMP (m=0)"

.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON \
  --improvement-task-assign-strategy M2M \
  --reallocation-task-method insertion \
  --output-file "$IRM2M_CMP" > "$LOGDIR/irm2m.log" 2>&1 &
IRM2M_PID=$!
echo "  irM2M  pid $IRM2M_PID -> $IRM2M_CMP"

wait "$M2M_PID"; M2M_RC=$?
wait "$CRM2M_PID"; CRM2M_RC=$?
wait "$IRM2M_PID"; IRM2M_RC=$?
echo "=== DONE: M2M(rc=$M2M_RC) crM2M(rc=$CRM2M_RC) irM2M(rc=$IRM2M_RC) ($(date)) ==="

echo "=== 3WAY 30min THROUGHPUT SUMMARY ==="
.venv/bin/python - "$M2M_CMP" "$CRM2M_CMP" "$IRM2M_CMP" <<'PY'
import json, sys
names = ["M2M", "crM2M", "irM2M"]
for name, fp in zip(names, sys.argv[1:]):
    try:
        d = json.load(open(fp))
    except FileNotFoundError:
        print(f"{name:8} (missing {fp} -- leg likely failed)"); continue
    t = int(d.get("timesteps_completed") or 0)
    r = int(d.get("total_completed_tasks") or 0)
    rate = r / (t / 60.0) if t else 0.0
    sh = int(d.get("total_completed_rearrangement_tasks") or 0)
    blk = int(d.get("outbound_buffer_placement_blocks") or 0)
    print(f"{name:8} {r:>6} real  {rate:>7.2f} tasks/min  | shuffles/inserts: {sh}  | buffer-blocks: {blk}  | ticks: {t}")
PY
echo "=== 3WAY 30min FINISHED ($(date)) ==="
