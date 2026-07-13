#!/usr/bin/env bash
# PARALLEL 30-minute (1800-tick) 2-way comparison: M2M vs crM2M, with the new
# crM2M return-margin term enabled.
#
# Same K=20 / mu=25 / 40-bot / 30%-inventory regime as run_4way_30min_parallel.sh,
# but only the two most relevant methods, and crM2M now requires each shuffle to
# finish with RETURN_MARGIN ticks of spare buffer-time before its slack pays off
# (--crm2m-return-margin). This targets the observed failure mode where a shuffle
# finishes just as a buffer slot opens, only for competing outbound arrivals to
# re-consume it before the agent returns to place.
#
# Both runs share the py_lns improvement strategy and therefore the same
# auto-generated output name, so each is given an explicit --output-file.
set -uo pipefail
cd /home/dcleeman/symbotic/M2M

MAP="$PWD/data/maps/study_small_restricted"
Q="$PWD/data/queues/uniform_tasking_buffer_30sku_inv30.txt"
INV="$PWD/data/initial_inventories/uniform_tasking_buffer_30sku_inv30_init_inventory.txt"

# crM2M return margin (ticks == seconds). Per Darren: require each shuffle to
# finish with ~10s of buffer headroom still remaining before it is judged
# worthwhile. With c_pp=4 and slack_cap=c_pp+cutoff/lambda=10.67, a 10-tick margin
# means only high-benefit / low-detour shuffles fired during buffer saturation
# will clear U>0 -- i.e. this deliberately keeps only the "safe" shuffles. This
# targets shuffles that finish right as a slot opens, only for competing outbound
# arrivals to re-consume it before the agent returns to place.
RETURN_MARGIN=10.0

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

M2M_CMP="data/raw_data/cmp_m2m_2way_delay30.json"
CRM2M_CMP="data/raw_data/cmp_crm2m_2way_delay30.json"

OUT_DIR="data/figures/twoway_delay_30min"
LOGDIR="/tmp/run_2way_delay"
mkdir -p "$LOGDIR"

echo "=== 2WAY DELAY30: launching M2M + crM2M (margin=$RETURN_MARGIN) ($(date)) ==="

.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON \
  --improvement-task-assign-strategy py_lns \
  --output-file "$M2M_CMP" > "$LOGDIR/m2m.log" 2>&1 &
M2M_PID=$!
echo "  M2M    pid $M2M_PID -> $M2M_CMP"

.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON \
  --improvement-task-assign-strategy py_lns \
  --enable-rearrangement --crm2m-lambda 1.5 --crm2m-detour-cutoff 10.0 \
  --crm2m-return-margin "$RETURN_MARGIN" \
  --output-file "$CRM2M_CMP" > "$LOGDIR/crm2m.log" 2>&1 &
CRM2M_PID=$!
echo "  crM2M  pid $CRM2M_PID -> $CRM2M_CMP"

wait "$M2M_PID"; M2M_RC=$?
wait "$CRM2M_PID"; CRM2M_RC=$?
echo "=== M2M (rc=$M2M_RC) + crM2M (rc=$CRM2M_RC) DONE -> figures ($(date)) ==="

if [[ $M2M_RC -eq 0 && $CRM2M_RC -eq 0 ]]; then
  .venv/bin/python GT_grid_world/scripts/plot_crm2m_compare.py "$M2M_CMP" "$CRM2M_CMP" "$OUT_DIR"
  .venv/bin/python GT_grid_world/scripts/plot_crm2m_rearrangement.py "$CRM2M_CMP" "$OUT_DIR" --tag crm2m
  echo "=== 2-WAY DELAY FIGURES READY in $OUT_DIR ==="
else
  echo "!!! M2M or crM2M failed; skipping figures. Check $LOGDIR."
fi

echo "=== 2WAY DELAY30 THROUGHPUT SUMMARY ==="
.venv/bin/python - "$M2M_CMP" "$CRM2M_CMP" <<'PY'
import json, sys
names = ["M2M", "crM2M"]
for name, fp in zip(names, sys.argv[1:]):
    try:
        d = json.load(open(fp))
    except FileNotFoundError:
        print(f"{name:8} (missing {fp})"); continue
    t = int(d.get("timesteps_completed") or 0)
    r = int(d.get("total_completed_tasks") or 0)
    rate = r / (t / 60.0) if t else 0.0
    sh = int(d.get("total_completed_rearrangement_tasks") or 0)
    blk = int(d.get("outbound_buffer_placement_blocks") or 0)
    print(f"{name:8} {r:>6} real  {rate:>7.2f} tasks/min  | shuffles: {sh}  | buffer-blocks: {blk}")
PY
echo "=== 2WAY DELAY30 FINISHED ($(date)) ==="
