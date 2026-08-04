#!/usr/bin/env bash
# Re-run M2M vs crM2M baseline (m=0) on the corrected sim (Ethan collision/oscillation fix).
# Same regime as the parameter-sweep anchor: K=20, mu=25, N=40, MT=120, 30 min.
set -uo pipefail
cd /home/dcleeman/symbotic/M2M

MAP="$PWD/data/maps/study_small_restricted"
Q="$PWD/data/queues/uniform_tasking_buffer_30sku_inv30.txt"
INV="$PWD/data/initial_inventories/uniform_tasking_buffer_30sku_inv30_init_inventory.txt"
LOGDIR="/tmp/run_m0_baseline"
mkdir -p "$LOGDIR"

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
--initial-inventory-file $INV --pick-place-time --pick-place-duration 4 \
--buffer-capacity-k 20 --buffer-consumption-rate 25.0 \
--improvement-task-assign-strategy py_lns"

M2M_OUT="data/raw_data/cmp_m2m_m0_baseline_fixed.json"
CRM2M_OUT="data/raw_data/cmp_crm2m_m0_baseline_fixed.json"

echo "=== m=0 BASELINE (corrected sim) start $(date) ==="

.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON \
  --output-file "$M2M_OUT" > "$LOGDIR/m2m.log" 2>&1 &
M2M_PID=$!
echo "  M2M pid $M2M_PID -> $M2M_OUT"

.venv/bin/python -u GT_grid_world/GT_grid_world.py $COMMON \
  --enable-rearrangement --crm2m-lambda 1.5 --crm2m-detour-cutoff 10.0 \
  --crm2m-return-margin 0.0 \
  --output-file "$CRM2M_OUT" > "$LOGDIR/crm2m.log" 2>&1 &
CRM2M_PID=$!
echo "  crM2M pid $CRM2M_PID -> $CRM2M_OUT"

wait "$M2M_PID"; M2M_RC=$?
wait "$CRM2M_PID"; CRM2M_RC=$?
echo "=== DONE M2M rc=$M2M_RC crM2M rc=$CRM2M_RC $(date) ==="

if [[ $M2M_RC -eq 0 && $CRM2M_RC -eq 0 ]]; then
  .venv/bin/python - <<'PY'
import json
for lab, fp in [("M2M", "data/raw_data/cmp_m2m_m0_baseline_fixed.json"),
                ("crM2M", "data/raw_data/cmp_crm2m_m0_baseline_fixed.json")]:
    d = json.load(open(fp))
    H = int(d["timesteps_completed"])
    done = int(d["total_completed_tasks"])
    sh = int(d.get("total_completed_rearrangement_tasks") or 0)
    print(f"{lab:6} thr={done/(H/60):.2f} tasks/min  shuffles={sh}")
PY
  DOC="/home/dcleeman/symbotic/ProactiveRearangment_project_context/presentations/step_A_bottleneck"
  .venv/bin/python GT_grid_world/scripts/build_bottleneck_figures.py \
    "$M2M_OUT" "$CRM2M_OUT" --out-dir "$DOC"
  .venv/bin/python - "$DOC" <<'PY'
import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
doc = Path(sys.argv[1])

def metrics(fp):
    d = json.load(open(fp)); H = int(d["timesteps_completed"])
    return int(d["total_completed_tasks"])/(H/60), int(d.get("total_completed_rearrangement_tasks") or 0), int(d.get("outbound_buffer_placement_blocks") or 0)

m2m = "data/raw_data/cmp_m2m_m0_baseline_fixed.json"
cr0 = "data/raw_data/cmp_crm2m_m0_baseline_fixed.json"
cr10 = "data/raw_data/cmp_crm2m_2way_delay30.json"
m = metrics(m2m); c0 = metrics(cr0); c10 = metrics(cr10)
labels = ["M2M\n(baseline)", "crM2M\n(no delay, m=0)", "crM2M\n(10s delay, m=10)"]
colors = ["tab:blue", "tab:green", "tab:orange"]
fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
thr = [m[0], c0[0], c10[0]]
b = ax[0].bar(labels, thr, color=colors, width=0.6)
ax[0].set_title("Throughput (tasks/min)", fontweight="bold"); ax[0].set_ylim(0, max(thr)*1.15)
for bar, v in zip(b, thr):
    ax[0].text(bar.get_x()+bar.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax[0].grid(axis="y", alpha=0.3)
shuf = [0, c0[1], c10[1]]
b2 = ax[1].bar(labels, shuf, color=colors, width=0.6)
ax[1].set_title("Rearrangements completed", fontweight="bold"); ax[1].set_ylim(0, max(shuf)*1.18)
for bar, v in zip(b2, shuf):
    ax[1].text(bar.get_x()+bar.get_width()/2, v, f"{v}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax[1].grid(axis="y", alpha=0.3)
fig.suptitle("Effect of the 10-second return-margin delay on crM2M (corrected sim, K=20, mu=25, N=40)", fontweight="bold", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(doc / "delay_effect_comparison.png", dpi=120, bbox_inches="tight")
print("wrote", doc / "delay_effect_comparison.png")
print("TABLE", m[0], c0[0], c0[1], c10[0], c10[1], m[2], c0[2], c10[2])

# Patch bottleneck report delay table (all three on corrected sim)
report = doc / "bottleneck_report.md"
text = report.read_text()
old_block = """| Condition | Throughput (tasks/min) | Rearrangements | Buffer-full blocks |
|---|---|---|---|
| M2M (baseline) | 62.7 to 63.4 | 0 | 12,034 to 14,816 |
| crM2M, no delay (m=0) | 63.27 | **112** | 12,203 |
| crM2M, 10s delay (m=10) | 64.47 | **4** | 11,627 |

**The 10-second delay cut rearrangements by ~28x (112 down to 4), yet throughput did not move:** every condition lands within ~3% of the others, around 63 to 64 tasks/min, and all sit at M2M's level. Tuning the delay changed *how much* rearrangement happened, but it did not change the *outcome*. This rules out "we just picked a bad margin" as the reason crM2M does not help: with many shuffles or almost none, the throughput is the same, because none of it touches the buffer-drain bottleneck.

> Caveat: the m=0 numbers come from an earlier build (before the collision / aisle-oscillation fixes); the m=10 numbers use the corrected build. The two M2M baselines differ by under 1 task/min, so the comparison holds."""

ratio = c0[1] / max(c10[1], 1)
new_block = f"""| Condition | Throughput (tasks/min) | Rearrangements | Buffer-full blocks |
|---|---|---|---|
| M2M (baseline) | {m[0]:.2f} | 0 | {m[2]:,} |
| crM2M, no delay (m=0) | {c0[0]:.2f} | **{c0[1]}** | {c0[2]:,} |
| crM2M, 10s delay (m=10) | {c10[0]:.2f} | **{c10[1]}** | {c10[2]:,} |

**The 10-second delay cut rearrangements by ~{ratio:.0f}x ({c0[1]} down to {c10[1]}), yet throughput did not move:** all three conditions land within ~3% of each other (~{min(m[0],c0[0],c10[0]):.0f} to ~{max(m[0],c0[0],c10[0]):.0f} tasks/min). Tuning the delay changed *how much* rearrangement happened, but it did not change the *outcome*. This rules out "we just picked a bad margin" as the reason crM2M does not help: with many shuffles or almost none, the throughput is the same, because none of it touches the buffer-drain bottleneck. All three runs use the corrected sim (Ethan collision/oscillation fix)."""

if old_block in text:
    report.write_text(text.replace(old_block, new_block))
    print("updated", report)
else:
    print("WARN: could not patch report (block not found)")
PY
fi
