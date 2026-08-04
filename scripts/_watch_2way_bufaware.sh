#!/usr/bin/env bash
# Watcher: as soon as the buffer-aware M2M (phase 1) and crM2M (phase 2) runs
# have both written their cmp JSONs, generate the 2-way M2M-vs-crM2M figure set
# into data/figures/bufaware/ (separate dir so it never clobbers the existing
# non-buffer crm2m-vs-m2m figures). The main run_4way_buffer_aware.sh keeps going
# and produces the full 4-way figure at the end.
set -uo pipefail
cd /home/dcleeman/symbotic/M2M

M2M_CMP="data/raw_data/cmp_m2m_bufaware_40bot_30pct_60min.json"
CRM2M_CMP="data/raw_data/cmp_crm2m_bufaware_40bot_30pct_60min.json"
OUT_DIR="data/figures/bufaware"

DEADLINE=$(( $(date +%s) + 6*3600 ))   # give up after 6 h
while :; do
  if [[ -f "$M2M_CMP" && -f "$CRM2M_CMP" ]]; then
    echo "=== 2WAY WATCH: both cmp files present, plotting M2M vs crM2M ==="
    .venv/bin/python GT_grid_world/scripts/plot_crm2m_compare.py \
      "$M2M_CMP" "$CRM2M_CMP" "$OUT_DIR"
    echo "=== 2WAY WATCH DONE: $OUT_DIR/throughput_rolling_crm2m-vs-m2m.png ==="
    break
  fi
  if (( $(date +%s) > DEADLINE )); then
    echo "=== 2WAY WATCH: timed out after 6 h waiting for cmp files ==="
    exit 1
  fi
  sleep 60
done
