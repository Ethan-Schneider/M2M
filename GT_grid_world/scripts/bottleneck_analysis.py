#!/usr/bin/env python3
"""Bottleneck decomposition for M2M vs crM2M runs.

Given one or more run JSONs, quantify *which* constraint limits throughput so we
can explain why rearrangement (which shortens future retrieval travel) does or
does not move throughput. Prints a comparison table and writes a 3-panel figure:
buffer occupancy, bot-idle fraction, and rolling mean retrieval (start->pick)
distance over time.

Usage:
    python bottleneck_analysis.py LABEL1=run1.json LABEL2=run2.json --out out.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATUS_FREE = 0


def _as_series(obj, horizon):
    """Coerce a dict{tick:val} or list into a dense float array of length horizon."""
    s = np.zeros(horizon, dtype=float)
    if isinstance(obj, dict):
        for k, v in obj.items():
            t = int(k)
            if 0 <= t < horizon:
                s[t] = float(v)
    elif isinstance(obj, list):
        for t, v in enumerate(obj[:horizon]):
            s[t] = float(v) if v is not None else 0.0
    return s


def _rolling(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def analyze(label, path):
    d = json.load(open(path))
    H = int(d.get("timesteps_completed") or d.get("time_horizon") or 1800)
    K = int(d.get("buffer_capacity_k") or 0)
    N = int(d.get("num_robots") or 0)

    completed = int(d.get("total_completed_tasks") or 0)
    thr = completed / (H / 60.0) if H else 0.0
    shuffles = int(d.get("total_completed_rearrangement_tasks") or 0)

    # --- Buffer binding: the buffer hovers at K-1/K under saturation and only
    # rarely records the exact cap (level is sampled post-drain), so "near cap"
    # (>= K-1) and mean utilisation are the honest saturation signals. ---
    buf = _as_series(d.get("output_buffer_level_per_timestep", {}), H)
    buf_near_cap = float(np.mean(buf >= (K - 1))) if K else float("nan")
    buf_util = float(np.mean(buf) / K) if K else float("nan")
    blocks = _as_series(d.get("buffer_blocks_per_timestep", {}), H)
    block_rate = float(np.mean(blocks))  # avg buffer-full rejections per tick
    total_blocks = int(d.get("outbound_buffer_placement_blocks") or 0)

    # --- Bot "waiting": fraction of bot-ticks spent stationary (not moving).
    # This captures bots frozen waiting to place at a full buffer / in pick-place
    # service, i.e. the slack that a drain-bound system wastes. STATUS_FREE
    # (no task at all) is tracked separately as true idleness. ---
    stationary = _as_series(d.get("stationary_robots", []), H)
    wait_frac = float(np.mean(stationary) / N) if N else float("nan")
    statuses = d.get("agent_statuses_per_timestep", [])
    idle_per_tick = np.zeros(H)
    if statuses:
        for t in range(min(H, len(statuses))):
            row = statuses[t]
            if row:
                idle_per_tick[t] = sum(1 for s in row if s == STATUS_FREE) / len(row)
    idle_frac = float(np.mean(idle_per_tick)) if statuses else float("nan")
    wait_series = stationary / N if N else stationary

    # --- Retrieval travel: distance from agent start to the pickup cell ---
    s2p = d.get("actual_distance_start_to_pick", {})
    vals = [float(v) for v in s2p.values()] if isinstance(s2p, dict) else [float(v) for v in s2p]
    mean_retrieval = float(np.mean(vals)) if vals else float("nan")
    # pick->place (fulfilment leg)
    p2p = d.get("actual_distance_of_task_from_pick_to_place", {})
    pvals = [float(v) for v in p2p.values()] if isinstance(p2p, dict) else [float(v) for v in p2p]
    mean_fulfil = float(np.mean(pvals)) if pvals else float("nan")

    # rolling retrieval distance over time, indexed by completion tick
    comp_ts = d.get("task_completion_timestamps", {})
    retr_by_tick = np.full(H, np.nan)
    counts = np.zeros(H)
    accum = np.zeros(H)
    if isinstance(comp_ts, dict) and isinstance(s2p, dict):
        for tid, tick in comp_ts.items():
            t = int(tick)
            if 0 <= t < H and tid in s2p:
                accum[t] += float(s2p[tid])
                counts[t] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        retr_by_tick = np.where(counts > 0, accum / np.maximum(counts, 1), np.nan)

    pf = int(d.get("Number of Path Plan Fails") or 0)

    return {
        "label": label, "H": H, "K": K, "N": N,
        "throughput": thr, "completed": completed, "shuffles": shuffles,
        "buf_util": buf_util, "buf_near_cap": buf_near_cap,
        "block_rate": block_rate, "total_blocks": total_blocks,
        "wait_frac": wait_frac, "idle_frac": idle_frac,
        "mean_retrieval": mean_retrieval, "mean_fulfil": mean_fulfil,
        "pf": pf, "buf_series": buf, "wait_series": wait_series,
        "retr_by_tick": retr_by_tick,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="LABEL=path.json")
    ap.add_argument("--out", default="bottleneck.png")
    ap.add_argument("--roll", type=int, default=60)
    args = ap.parse_args()

    results = []
    for spec in args.runs:
        label, path = spec.split("=", 1)
        results.append(analyze(label, path))

    # ---- table ----
    cols = [
        ("throughput", "thru/min", "{:.2f}"),
        ("completed", "done", "{:d}"),
        ("shuffles", "shuffle", "{:d}"),
        ("buf_util", "buf util%", "{:.1%}"),
        ("buf_near_cap", "buf>=K-1%", "{:.1%}"),
        ("block_rate", "blk/tick", "{:.2f}"),
        ("wait_frac", "bot wait%", "{:.1%}"),
        ("mean_retrieval", "retr dist", "{:.2f}"),
        ("mean_fulfil", "fulfil dist", "{:.2f}"),
        ("pf", "PBS fails", "{:d}"),
    ]
    hdr = f"{'run':10}" + "".join(f"{h:>11}" for _, h, _ in cols)
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        line = f"{r['label']:10}"
        for key, _, fmt in cols:
            line += f"{fmt.format(r[key]):>11}"
        print(line)

    # ---- figure ----
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    x = None
    for r in results:
        H = r["H"]
        x = np.arange(H)
        axes[0].plot(x, r["buf_series"], label=f"{r['label']} (K={r['K']})", lw=1.2)
        axes[1].plot(x, _rolling(r["wait_series"], args.roll), label=r["label"], lw=1.4)
        y = r["retr_by_tick"]
        m = ~np.isnan(y)
        axes[2].plot(x[m], _rolling(np.nan_to_num(y), args.roll)[m], label=r["label"], lw=1.2)

    if results:
        axes[0].axhline(results[0]["K"], ls="--", c="k", alpha=0.5, label=f"capacity K={results[0]['K']}")
    axes[0].set_ylabel("output buffer level")
    axes[0].set_title("Buffer occupancy over time (pinned at K => drain-bound)")
    axes[0].legend(fontsize=8)

    axes[1].set_ylabel(f"bot stationary fraction ({args.roll}-tick roll)")
    axes[1].set_title("Bot waiting fraction (bots frozen at full buffer / in service)")
    axes[1].legend(fontsize=8)

    axes[2].set_ylabel(f"mean start->pick dist ({args.roll}-tick roll)")
    axes[2].set_title("Retrieval travel over time (what rearrangement is supposed to reduce)")
    axes[2].set_xlabel("timestep")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=120)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
