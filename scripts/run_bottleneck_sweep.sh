#!/usr/bin/env bash
# Bottleneck sweep (B1 buffer drain/capacity, B2 offered load, B3 bot count).
#
# Goal: move the binding constraint off the buffer drain (mu) and see whether
# crM2M ever separates from M2M on throughput. Step A showed the K=20/mu=25/N=40
# regime is drain-bound (buffer ~90% full, 42% of bot-time spent waiting to
# place), which masks the ~2-6% travel improvement crM2M produces.
#
# crM2M uses margin m=0 here (max shuffle activity) to give rearrangement its
# best chance; lambda=1.5, cutoff=10 as before.
#
# All configs share the same demand-driven queue + initial inventory; the only
# knobs varied are --buffer-consumption-rate (mu), --buffer-capacity-k (K),
# --num-robots (N) and --max-tasks (MT, the WIP cap = offered-load lever, since
# release is backpressured by len(J) < MT).
set -uo pipefail
cd /home/dcleeman/symbotic/M2M

MAP="$PWD/data/maps/study_small_restricted"
Q="$PWD/data/queues/uniform_tasking_buffer_30sku_inv30.txt"
INV="$PWD/data/initial_inventories/uniform_tasking_buffer_30sku_inv30_init_inventory.txt"

OUT_DIR="data/raw_data/sweep"
LOGDIR="/tmp/run_sweep"
mkdir -p "$OUT_DIR" "$LOGDIR"

MAX_CONCURRENCY="${MAX_CONCURRENCY:-10}"

BASE="--seed 0 --time-horizon 1800 \
--initial-task-assign-strategy fast_greedy --path-planning-strategy pbs \
--initial-inventory 30.0 --frequency 0.25 --inbound-outbound-ratio 1.0 \
--num-skus 30 --weight-init-method uniform --map $MAP \
--cost-calculation-method shortest_path --removal-operator shaw \
--repair-operator greedy --acceptance-function simulated_annealing \
--T-0 1.0 --alpha 0.99 --deadline-generation-method normal --deadline-offset 180 \
--deadline-weight 0.25 --base-cost-weight 1.0 --sku-distribution-weight 0.0 \
--agent-unallocated-penalty 5.0 --W 300 --B 60 \
--task-gen-strategy feedback_control --use-precomputed-queue --queue-file $Q \
--initial-inventory-file $INV --pick-place-time --pick-place-duration 4 \
--improvement-task-assign-strategy py_lns"

CRM2M_FLAGS="--enable-rearrangement --crm2m-lambda 1.5 --crm2m-detour-cutoff 10.0 --crm2m-return-margin 0.0"

# config: NAME  K  MU  N  MT   (baseline is N40_K20_mu25_mt120)
CONFIGS=(
  "N40_K20_mu25_mt120  20 25  40 120"   # baseline / shared anchor
  "N40_K20_mu40_mt120  20 40  40 120"   # B1 mu sweep
  "N40_K20_mu60_mt120  20 60  40 120"
  "N40_K20_mu120_mt120 20 120 40 120"
  "N40_K40_mu25_mt120  40 25  40 120"   # B1 K sweep
  "N40_K80_mu25_mt120  80 25  40 120"
  "N40_K20_mu25_mt40   20 25  40 40"    # B2 offered-load (WIP cap)
  "N40_K20_mu25_mt80   20 25  40 80"
  "N20_K20_mu25_mt120  20 25  20 120"   # B3 bot count
  "N30_K20_mu25_mt120  20 25  30 120"
  "N60_K20_mu25_mt120  20 25  60 120"
)

launch () {
  local method="$1" name="$2" K="$3" MU="$4" N="$5" MT="$6"
  local out="$OUT_DIR/${method}_${name}.json"
  local log="$LOGDIR/${method}_${name}.log"
  local extra=""
  [[ "$method" == "crm2m" ]] && extra="$CRM2M_FLAGS"
  .venv/bin/python -u GT_grid_world/GT_grid_world.py $BASE \
    --num-robots "$N" --max-tasks "$MT" \
    --buffer-capacity-k "$K" --buffer-consumption-rate "$MU" \
    $extra --output-file "$out" > "$log" 2>&1 &
  echo "  launched $method $name (pid $!) -> $out"
}

echo "=== BOTTLENECK SWEEP start ($(date)) | concurrency=$MAX_CONCURRENCY ==="
running=0
for cfg in "${CONFIGS[@]}"; do
  read -r name K MU N MT <<< "$cfg"
  for method in m2m crm2m; do
    launch "$method" "$name" "$K" "$MU" "$N" "$MT"
    running=$((running + 1))
    if (( running >= MAX_CONCURRENCY )); then
      wait -n
      running=$((running - 1))
    fi
  done
done
wait
echo "=== ALL SWEEP RUNS DONE ($(date)) ==="

echo "=== SWEEP THROUGHPUT TABLE ==="
.venv/bin/python GT_grid_world/scripts/plot_sweep.py "$OUT_DIR" \
  --out-dir data/figures/bottleneck_sweep
echo "=== SWEEP FIGURES READY in data/figures/bottleneck_sweep ($(date)) ==="
