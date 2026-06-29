"""Generate the six canonical baseline visuals for a controlled M2M vs
M2M vs crM2M A/B comparison.

This reuses the exact metric definitions from ``plot_baselines.py`` (the
team's established figure set) but, because the A/B differs only by
``--enable-rearrangement`` (same density / robot count), it overlays the
two conditions as the two series on a single panel each instead of
faceting by density. The six figure types -- and only these six -- are:

  1. throughput_rolling                -- tasks/min, rolling mean +/- 1 std
  2. sku_spread_trajectories           -- SKU Spread (EZC) vs time
  3. bot_utilization_trajectories      -- productive bot fraction vs time
  4. computation_time_trajectories     -- TA + PF runtime per tick (log y)
  5. cumulative_tardiness              -- cumulative task tardiness (s)
  6. cumulative_tardy_tasks            -- cumulative count of tardy tasks

Usage:
    python scripts/plot_crm2m_compare.py BASELINE.json CRM2M.json [OUT_DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_baselines as pb

# M2M vs crM2M series styling.
SERIES = [
    ("M2M", pb_color_baseline := "#1f77b4"),
    ("crM2M", pb_color_crm2m := "#d62728"),
]

FILE_TAG = "crm2m-vs-m2m"


def _load(fp: Path) -> dict:
    import json
    with fp.open() as f:
        return json.load(f)


def _xmax(runs: list[dict]) -> int:
    return max((int(r.get("timesteps_completed") or 0) for r in runs), default=500) or 500


def plot_throughput_rolling(runs, labels, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    DEFAULT_WINDOW_MINUTES = 5
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
        ax.fill_between(t, lower, mean + std, color=color, alpha=0.18, linewidth=0)
        ax.plot(t, mean, label=label, color=color, lw=1.8)
        y_max = max(y_max, float((mean + std).max()) if mean.size else 0.0)
    ax.set_xlim(0, _xmax(runs))
    ax.set_ylim(0, max(10.0, y_max * 1.05))
    ax.set_xlabel("Simulation timestep")
    ax.set_ylabel("Throughput (tasks / min)")
    # Throughput sits high (~100-130/min) so the upper-right collides with the
    # lines; the lower-right corner is empty -> park the legend there.
    ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Throughput (tasks / min) rolling mean +/- 1 std -- M2M vs crM2M\n"
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
    fig.suptitle("SKU Spread (EZC) trajectories -- M2M vs crM2M (lower is better)",
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
    fig.suptitle("Productive bot fraction (11-step rolling) -- M2M vs crM2M",
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
        "Per-timestep computation time -- M2M vs crM2M (log y)\n"
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
        "Cumulative task tardiness -- M2M vs crM2M\n"
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
        "Cumulative tardy tasks -- M2M vs crM2M\n"
        "tasks whose completion timestep exceeds their deadline",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("crm2m", type=Path)
    ap.add_argument("out_dir", type=Path, nargs="?", default=Path("data/figures"))
    args = ap.parse_args()

    pb._set_paper_style()
    runs = [_load(args.baseline), _load(args.crm2m)]
    labels = SERIES
    args.out_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        ("throughput_rolling", plot_throughput_rolling),
        ("sku_spread_trajectories", plot_sku_spread),
        ("bot_utilization_trajectories", plot_bot_utilization),
        ("computation_time_trajectories", plot_computation_time),
        ("cumulative_tardiness", plot_cumulative_tardiness),
        ("cumulative_tardy_tasks", plot_cumulative_tardy_tasks),
    ]
    for name, fn in figures:
        out = args.out_dir / f"{name}_{FILE_TAG}.png"
        fn(runs, labels, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
