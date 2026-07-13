"""Full 4-way comparison figure set: crM2M against the three baselines
(M2M, LNS-PBS, HBH+MLA*) for every *shared* metric.

This produces the same metric definitions and styling as
``plot_crm2m_compare.py`` (the team's canonical A/B set) but overlays four
conditions instead of two, and writes to dedicated ``*_4way`` filenames.

Rearrangement-specific figures (benefit / detour cost / utility) are
deliberately EXCLUDED here: only crM2M rearranges, so those charts are
generated separately by ``plot_crm2m_rearrangement.py`` for the crM2M run
alone. The shared metrics produced here are:

  1. throughput_rolling            -- tasks/min, rolling mean +/- 1 std
  2. sku_spread_trajectories       -- SKU Spread (EZC) vs time
  3. bot_utilization_trajectories  -- productive bot fraction vs time
  4. computation_time_trajectories -- TA + PF runtime per tick (log y)
  5. cumulative_tardiness          -- cumulative task tardiness (s)
  6. cumulative_tardy_tasks        -- cumulative count of tardy tasks
  7. buffer_state                  -- shared output buffer level vs time
  8. rolling_task_duration         -- rolling per-task service time vs time
  9. cumulative_buffer_blocks      -- cumulative outbound buffer-full blocks

All four runs must share the same config (same horizon, bots, inventory,
queue, deadlines) so the only difference is the allocation method.

Usage:
    python scripts/plot_4way_all.py M2M.json CRM2M.json LNSPBS.json HBH.json [OUT_DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_baselines as pb

# (legend label, colour) in the positional order the run JSONs are passed.
SERIES = [
    ("M2M", "#1f77b4"),       # blue
    ("crM2M", "#d62728"),     # red
    ("LNS-PBS", "#2ca02c"),   # green
    ("HBH+MLA*", "#ff7f0e"),  # orange
]

FILE_TAG = "4way"
TITLE_SUFFIX = "crM2M vs baselines (M2M, LNS-PBS, HBH+MLA*)"

DEFAULT_WINDOW_MINUTES = 5


def _load(fp: Path) -> dict:
    with fp.open() as f:
        return json.load(f)


def _xmax(runs) -> int:
    return max((int(r.get("timesteps_completed") or 0) for r in runs), default=500) or 500


def _rolling_task_duration(run: dict):
    """Return (times, rolling_mean, rolling_std) of per-task service time
    (release -> completion, seconds) ordered by completion timestep."""
    service = run.get("service_times") or {}
    completions = run.get("task_completion_timestamps") or {}
    pairs = []
    for tid, dur in service.items():
        ct = completions.get(str(tid))
        if ct is None:
            continue
        pairs.append((int(ct), float(dur)))
    if not pairs:
        return np.array([]), np.array([]), np.array([])
    pairs.sort()
    times = np.array([p[0] for p in pairs], dtype=float)
    durs = np.array([p[1] for p in pairs], dtype=float)
    window = max(5, durs.size // 20)
    if durs.size < window:
        return times, durs, np.zeros_like(durs)
    kernel = np.ones(window) / window
    mean = np.convolve(durs, kernel, mode="valid")
    sq = np.convolve(durs ** 2, kernel, mode="valid")
    std = np.sqrt(np.clip(sq - mean ** 2, 0.0, None))
    offset = window // 2
    t = times[offset: offset + mean.size]
    return t, mean, std


def plot_throughput_rolling(runs, labels, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_max = 0.0
    for run, (label, color) in zip(runs, labels):
        centers, rate, n_bins = pb._per_minute_throughput(run)
        if n_bins == 0:
            continue
        window = min(DEFAULT_WINDOW_MINUTES, max(2, n_bins // 3))
        if n_bins < window:
            t, mean, std = centers, rate, np.zeros_like(rate)
        else:
            offset_idx, mean, std = pb._rolling_mean_std(rate, window)
            t = centers[offset_idx[0]: offset_idx[0] + mean.size]
        lower = np.clip(mean - std, 0.0, None)
        ax.fill_between(t, lower, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.plot(t, mean, label=label, color=color, lw=1.8)
        y_max = max(y_max, float((mean + std).max()) if mean.size else 0.0)
    ax.set_xlim(0, _xmax(runs))
    ax.set_ylim(0, max(10.0, y_max * 1.05))
    ax.set_xlabel("Simulation timestep")
    ax.set_ylabel("Throughput (tasks / min)")
    ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(
        f"Throughput (tasks / min) rolling mean +/- 1 std -- {TITLE_SUFFIX}\n"
        "1-minute bins; rolling window = min(5 min, n_bins/3); std = minute-to-minute jitter",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_sku_spread(runs, labels, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run, (label, color) in zip(runs, labels):
        spread = np.asarray(run.get("sku_spread_per_timestep") or [], dtype=float)
        if spread.size == 0:
            continue
        ax.plot(np.arange(spread.size), spread, label=label, color=color, lw=1.6)
    ax.set_xlim(0, _xmax(runs))
    ax.set_xlabel("Simulation timestep")
    ax.set_ylabel("SKU Spread (EZC)")
    ax.legend(loc="lower right")
    fig.suptitle(f"SKU Spread (EZC) trajectories -- {TITLE_SUFFIX} (lower is better)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_bot_utilization(runs, labels, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run, (label, color) in zip(runs, labels):
        t, util = pb._utilization_series(run)
        if util.size == 0:
            continue
        ax.plot(t, util, label=label, color=color, lw=1.6)
    ax.set_xlim(0, _xmax(runs))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Simulation timestep")
    ax.set_ylabel("Fraction of bots productive (status > 0)")
    ax.legend(loc="best")
    fig.suptitle(f"Productive bot fraction (11-step rolling) -- {TITLE_SUFFIX}",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_computation_time(runs, labels, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    row_specs = [
        ("Task Allocation Runtime", "Task Allocation runtime (s / tick, log)",
         pb._TA_PLOT_FLOOR, pb._TA_PLOT_CEIL, pb._TA_YLIM),
        ("Path Planning Runtimes", "Path Planning runtime (s / tick, log)",
         pb._PF_PLOT_FLOOR, None, pb._PF_YLIM),
    ]
    for ax, (json_key, ylabel, floor, ceil, ylim) in zip(axes, row_specs):
        for run, (label, color) in zip(runs, labels):
            arr = np.asarray(run.get(json_key) or [], dtype=float)
            if arr.size == 0:
                continue
            plot_arr = np.where(arr > 0, np.maximum(arr, floor), np.nan)
            if ceil is not None:
                plot_arr = np.minimum(plot_arr, ceil)
            stat = (f"{label}  (avg={arr.mean():.2f}, med={np.median(arr):.2f}, "
                    f"std={arr.std():.2f})")
            ax.plot(np.arange(arr.size), plot_arr, label=stat, color=color, lw=1.0, alpha=0.7)
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
    axes[1].set_xlabel("Simulation timestep")
    axes[0].set_xlim(0, _xmax(runs))
    fig.suptitle(
        f"Per-timestep computation time -- {TITLE_SUFFIX} (log y)\n"
        "TA panel rendered with a 10 s ceiling for visual stability",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_tardiness(runs, labels, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_max = 0.0
    for run, (label, color) in zip(runs, labels):
        n = int(run.get("timesteps_completed") or 0)
        if n <= 0:
            continue
        completions = run.get("task_completion_timestamps") or {}
        tardy = np.zeros(n, dtype=float)
        for tid, excess in pb._per_task_tardiness(run).items():
            ct = completions.get(str(tid))
            if ct is None:
                continue
            t = int(ct)
            if 0 <= t < n:
                tardy[t] += excess
        cum = np.cumsum(tardy)
        ax.plot(np.arange(n), cum, label=label, color=color, lw=1.6)
        y_max = max(y_max, float(cum[-1]) if cum.size else 0.0)
    ax.set_xlim(0, _xmax(runs))
    ax.set_ylim(0, max(10.0, y_max * 1.05))
    ax.set_xlabel("Simulation timestep")
    ax.set_ylabel("Cumulative task tardiness (s)")
    ax.legend(loc="upper left")
    fig.suptitle(
        f"Cumulative task tardiness -- {TITLE_SUFFIX}\n"
        "per-task tardiness = max(0, completion_t - deadline_t)",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_tardy_tasks(runs, labels, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_max = 0
    for run, (label, color) in zip(runs, labels):
        n = int(run.get("timesteps_completed") or 0)
        if n <= 0:
            continue
        completions = run.get("task_completion_timestamps") or {}
        cnt = np.zeros(n, dtype=int)
        for tid in pb._per_task_tardiness(run):
            ct = completions.get(str(tid))
            if ct is None:
                continue
            t = int(ct)
            if 0 <= t < n:
                cnt[t] += 1
        cum = np.cumsum(cnt)
        ax.plot(np.arange(n), cum, label=label, color=color, lw=1.6)
        if cum.size:
            y_max = max(y_max, int(cum[-1]))
    ax.set_xlim(0, _xmax(runs))
    ax.set_ylim(0, max(5, int(y_max * 1.10) + 1))
    ax.set_xlabel("Simulation timestep")
    ax.set_ylabel("Cumulative tardy tasks")
    ax.legend(loc="upper left")
    fig.suptitle(
        f"Cumulative tardy tasks -- {TITLE_SUFFIX}\n"
        "tasks whose completion timestep exceeds their deadline",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_buffer_state(runs, labels, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cap = None
    for run, (label, color) in zip(runs, labels):
        levels = run.get("output_buffer_level_per_timestep") or {}
        if not levels:
            continue
        items = sorted((int(k), float(v)) for k, v in levels.items())
        t = np.array([i[0] for i in items], dtype=float)
        y = np.array([i[1] for i in items], dtype=float)
        ax.plot(t, y, label=label, color=color, lw=1.3)
        k = run.get("buffer_capacity_k")
        if k:
            cap = float(k)
    if cap:
        ax.axhline(cap, color="0.35", ls="--", lw=1.0, label=f"capacity K={cap:.0f}")
    ax.set_xlim(0, _xmax(runs))
    ax.set_ylim(0, (cap * 1.1 if cap else None))
    ax.set_xlabel("Simulation timestep")
    ax.set_ylabel("Output buffer level (items)")
    ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(
        f"Output buffer level over time -- {TITLE_SUFFIX}\n"
        "shared outbound buffer occupancy; a level pinned at K means outbound is buffer-bound",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_task_duration(runs, labels, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_max = 0.0
    for run, (label, color) in zip(runs, labels):
        t, mean, std = _rolling_task_duration(run)
        if mean.size == 0:
            continue
        ax.fill_between(t, np.clip(mean - std, 0.0, None), mean + std,
                        color=color, alpha=0.12, linewidth=0)
        ax.plot(t, mean, label=label, color=color, lw=1.8)
        y_max = max(y_max, float((mean + std).max()))
    ax.set_xlim(0, _xmax(runs))
    ax.set_ylim(0, max(10.0, y_max * 1.05))
    ax.set_xlabel("Simulation timestep")
    ax.set_ylabel("Task duration (s, release -> completion)")
    ax.legend(loc="best", fontsize=9)
    fig.suptitle(
        f"Rolling task duration -- {TITLE_SUFFIX}\n"
        "per-task service time = completion_t - release_t; window = max(5, n/20) tasks",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_buffer_blocks(runs, labels, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_max = 0.0
    for run, (label, color) in zip(runs, labels):
        n = int(run.get("timesteps_completed") or 0)
        blocks = run.get("buffer_blocks_per_timestep") or {}
        if n <= 0:
            continue
        per = np.zeros(n, dtype=float)
        for k, v in blocks.items():
            ti = int(k)
            if 0 <= ti < n:
                per[ti] += float(v)
        cum = np.cumsum(per)
        total = int(cum[-1]) if cum.size else 0
        ax.plot(np.arange(n), cum, label=f"{label} (total={total})", color=color, lw=1.6)
        y_max = max(y_max, float(cum[-1]) if cum.size else 0.0)
    ax.set_xlim(0, _xmax(runs))
    ax.set_ylim(0, max(5.0, y_max * 1.10))
    ax.set_xlabel("Simulation timestep")
    ax.set_ylabel("Cumulative buffer-full block events")
    ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(
        f"Cumulative outbound buffer-full blocks -- {TITLE_SUFFIX}\n"
        "each event = one agent-tick an outbound place was blocked by a full buffer",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("m2m", type=Path)
    ap.add_argument("crm2m", type=Path)
    ap.add_argument("lnspbs", type=Path)
    ap.add_argument("hbh", type=Path)
    ap.add_argument("out_dir", type=Path, nargs="?", default=Path("data/figures"))
    args = ap.parse_args()

    pb._set_paper_style()
    runs = [_load(args.m2m), _load(args.crm2m), _load(args.lnspbs), _load(args.hbh)]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        ("throughput_rolling", plot_throughput_rolling),
        ("sku_spread_trajectories", plot_sku_spread),
        ("bot_utilization_trajectories", plot_bot_utilization),
        ("computation_time_trajectories", plot_computation_time),
        ("cumulative_tardiness", plot_cumulative_tardiness),
        ("cumulative_tardy_tasks", plot_cumulative_tardy_tasks),
        ("buffer_state", plot_buffer_state),
        ("rolling_task_duration", plot_rolling_task_duration),
        ("cumulative_buffer_blocks", plot_cumulative_buffer_blocks),
    ]
    for name, fn in figures:
        out = args.out_dir / f"{name}_{FILE_TAG}.png"
        fn(runs, SERIES, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
