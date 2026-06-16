"""Generate the headline visuals for the M2M baseline sweep.

Reads every run JSON in ``data/raw_data/`` (one per (density, robots)
condition produced by ``run_baselines.sh``) and writes the figures listed
below to ``data/figures/``. Trajectory figures are faceted by inventory
density (3 panels: 30 / 60 / 90%) with one line per robot count.

Figures (current set, post June-15 metrics revision):

* ``throughput_rolling.png`` -- Rolling mean +/- 1 std of tasks completed
  per minute. Window = 10% of total simulation time. Replaces the
  previous static throughput heatmap; chosen on the routing team's
  request because the rolling view shows the steady-state behaviour
  rather than collapsing it into a single number.
* ``sku_spread_trajectories.png`` -- EZC (SKU Spread) vs simulation time.
  Lead headline metric for proactive rearrangement: lower-and-bounded
  EZC is the explicit goal of the work.
* ``bot_utilization_trajectories.png`` -- Fraction of bots in a
  productive state (carrying or en-route to pickup, derived from
  ``agent_statuses_per_timestep``) vs simulation time, 11-step rolling
  mean.
* ``computation_time_trajectories.png`` -- Per-timestep task-allocation
  runtime and per-timestep path-planning runtime. 2 rows (TA / PF) x 3
  density panels. Each panel annotates avg / median / std per robot
  count to back the team's "report avg, median, std" requirement.
* ``cumulative_tardiness.png`` -- Cumulative task tardiness (seconds)
  over simulation time, computed from ``completed_task_details``
  (deadline) and ``task_completion_timestamps``. Per-task tardiness =
  max(0, completion_t - deadline_t).
* ``cumulative_tardy_tasks.png`` -- Cumulative count of tasks completed
  past their deadline, over simulation time.

Removed in the same revision: ``throughput_heatmap.png`` (superseded by
the rolling view) and ``cumulative_completed_tasks.png`` (collapses to
y = mx + b at scale and adds no insight beyond the throughput plot).

These are intentionally exploratory baselines (single seed, single map,
10-min/condition cap), not publication-grade.

Usage:
    python scripts/plot_baselines.py [data/raw_data] [data/figures]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DENSITY_ORDER = [30.0, 60.0, 90.0]
ROBOT_PALETTE = {10: "#1f77b4", 25: "#ff7f0e", 40: "#2ca02c"}

# Map M2M's `improvement_task_assignment_strategy` to a human-readable
# baseline label. `c_lns` routes through the `external_algorithms/lns/`
# C++ submodule (whose README identifies it as LNS-PBS, Jiaoyang Li);
# `py_lns` routes through `py_lns_V2` over `construct_cost_elements`,
# which is M2M's native 4D-cost-tensor + LNS allocator.
METHOD_LABELS = {
    "c_lns": "LNS-PBS",
    "py_lns": "M2M (4D cost tensor + py_lns)",
}


def _set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "legend.frameon": False,
        }
    )


def _load_runs(raw_dir: Path) -> list[dict]:
    runs = []
    for fp in sorted(raw_dir.glob("*.json")):
        try:
            with fp.open() as f:
                runs.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return runs


def _index_by_condition(runs: list[dict]) -> dict[tuple[float, int], dict]:
    """Map (density, robots) -> latest run dict."""
    out: dict[tuple[float, int], dict] = {}
    for d in runs:
        density = float(d.get("initial_inventory"))
        robots = int(d.get("num_robots"))
        out[(density, robots)] = d
    return out


def _detect_method_label(runs: list[dict]) -> str:
    """Return a human-readable label for the allocator used in ``runs``.

    If every run uses the same ``improvement_task_assignment_strategy``,
    return its mapped label; otherwise return a 'mixed methods' tag and
    let the caller decide how to disambiguate. Falls back to whatever
    string is in the JSON if it's not in our mapping (so new allocators
    surface a readable name instead of silently mislabelling).
    """
    strategies = {r.get("improvement_task_assignment_strategy") for r in runs}
    strategies.discard(None)
    if not strategies:
        return "unknown method"
    if len(strategies) == 1:
        s = next(iter(strategies))
        return METHOD_LABELS.get(s, s)
    return "mixed methods (" + ", ".join(sorted(METHOD_LABELS.get(s, s) for s in strategies)) + ")"


def _ax_title_density(density: float) -> str:
    return f"Density = {density:.0f}%"


def _show_all_y_tick_labels(axes) -> None:
    """Force y-tick labels on every panel of a faceted figure.

    With ``sharey=True`` matplotlib hides y-tick labels on every panel
    except the leftmost one, which makes it look at-a-glance like each
    panel has its own scale (the routing team explicitly flagged this
    on the first cut of the visuals). Re-enabling ``labelleft`` on
    every axis keeps the *axis sharing* (numeric ranges still propagate
    via ``sharey``) but makes the shared scale visually obvious because
    every panel renders the same tick labels.
    """
    flat = axes.flat if hasattr(axes, "flat") else axes
    for ax in flat:
        ax.tick_params(labelleft=True)


def _global_xmax(by_cond: dict) -> int:
    """Maximum simulation length (in ticks) across all loaded conditions.

    Used to set a shared x-axis upper bound so cross-density panels are
    visually comparable. Falls back to a sensible default if the runs
    don't expose ``timesteps_completed``.
    """
    best = 0
    for run in by_cond.values():
        n = int(run.get("timesteps_completed") or 0)
        if n > best:
            best = n
    return best or 500


def plot_sku_spread(by_cond: dict, out_path: Path, method_label: str) -> None:
    """SKU Spread (EZC) trajectories with shared x and y axes.

    Cross-density y range is intentionally large (~120 to ~720) because
    EZC is count-weighted and scales with absolute SKU population. The
    shared axes are explicitly the routing team's preference: cross-panel
    visual compression is acceptable in exchange for an unambiguous
    cross-condition comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    y_min = float("inf")
    y_max = 0.0
    for ax, density in zip(axes, DENSITY_ORDER):
        for robots in sorted(ROBOT_PALETTE):
            run = by_cond.get((density, robots))
            if not run:
                continue
            spread = run.get("sku_spread_per_timestep") or []
            if not spread:
                continue
            arr = np.asarray(spread, dtype=float)
            t = np.arange(arr.size)
            ax.plot(
                t,
                arr,
                label=f"{robots} bots",
                color=ROBOT_PALETTE[robots],
                lw=1.6,
            )
            y_min = min(y_min, float(arr.min()))
            y_max = max(y_max, float(arr.max()))
        ax.set_title(_ax_title_density(density))
        ax.set_xlabel("Simulation timestep")
        if density == DENSITY_ORDER[0]:
            ax.set_ylabel("SKU Spread (EZC)")
        ax.legend(loc="lower right")
    if y_min < float("inf"):
        pad = max(5.0, (y_max - y_min) * 0.05)
        axes[0].set_ylim(max(0.0, y_min - pad), y_max + pad)
    axes[0].set_xlim(0, _global_xmax(by_cond))
    _show_all_y_tick_labels(axes)
    fig.suptitle(
        f"SKU Spread (EZC) trajectories -- {method_label} baseline (no rearrangement)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _per_tick_completions(run: dict) -> tuple[np.ndarray, int]:
    """Return per-timestep completion counts and the actual sim length.

    Reconstructs an array of length ``timesteps_completed`` where
    ``arr[t]`` is the number of tasks whose ``task_completion_timestamps``
    value equals ``t``. Tasks that completed after the run's actual
    horizon (defensive: should not happen) are clipped.
    """
    n_ticks = int(run.get("timesteps_completed") or 0)
    if n_ticks <= 0:
        return np.zeros(0, dtype=int), 0
    arr = np.zeros(n_ticks, dtype=int)
    ts = run.get("task_completion_timestamps") or {}
    for v in ts.values():
        t = int(v)
        if 0 <= t < n_ticks:
            arr[t] += 1
    return arr, n_ticks


def _rolling_window(n_ticks: int, frac: float = 0.10, min_window: int = 5) -> int:
    """Window length used for rolling stats (= ``frac`` of total ticks).

    ``min_window`` keeps very short runs (e.g. interrupted smoke tests)
    from collapsing the rolling stats to a single sample.
    """
    return max(min_window, int(round(frac * n_ticks)))


def _rolling_mean_std(rate: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centered rolling mean and population std over ``window`` samples.

    Uses ``np.convolve`` for the mean and a streaming
    ``E[X^2] - E[X]^2`` formulation for the std (clipped at zero to
    absorb floating-point negatives). Returns a time axis aligned to
    the centre of each window so the curves sit at the correct sim time.
    """
    if rate.size < window:
        return np.array([]), np.array([]), np.array([])
    kernel = np.ones(window) / window
    mean = np.convolve(rate, kernel, mode="valid")
    mean_sq = np.convolve(rate * rate, kernel, mode="valid")
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    std = np.sqrt(var)
    offset = (window - 1) // 2
    t = np.arange(offset, offset + mean.size)
    return t, mean, std


def plot_throughput_rolling(by_cond: dict, out_path: Path, method_label: str) -> None:
    """Rolling tasks/min (mean +/- 1 std), window = 10% of total sim time.

    Construction: per-tick instantaneous tasks/min = ``60 *
    completions_at_tick`` (since 1 sim tick == 1 simulated second).
    Rolling mean and population std are computed over the same centered
    window. The +/- 1 sigma fill is clipped at zero so the band never
    dips below the x-axis. Std band alpha is intentionally very low
    (0.06) because the within-window std is large for a Bernoulli-ish
    completion process and a heavier band visually overwhelms the means.
    Axes are shared across density panels to make cross-density
    comparison unambiguous.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    y_max = 0.0
    for ax, density in zip(axes, DENSITY_ORDER):
        for robots in sorted(ROBOT_PALETTE):
            run = by_cond.get((density, robots))
            if not run:
                continue
            counts, n_ticks = _per_tick_completions(run)
            if n_ticks == 0:
                continue
            rate = counts.astype(float) * 60.0
            window = _rolling_window(n_ticks)
            t, mean, std = _rolling_mean_std(rate, window)
            if t.size == 0:
                continue
            color = ROBOT_PALETTE[robots]
            lower = np.clip(mean - std, 0.0, None)
            upper = mean + std
            ax.fill_between(t, lower, upper, color=color, alpha=0.06, linewidth=0)
            ax.plot(t, mean, label=f"{robots} bots", color=color, lw=1.8)
            y_max = max(y_max, float(upper.max()))
        ax.set_title(_ax_title_density(density))
        ax.set_xlabel("Simulation timestep")
        if density == DENSITY_ORDER[0]:
            ax.set_ylabel("Throughput (tasks / min)")
        ax.legend(loc="upper right", fontsize=9)
    axes[0].set_xlim(0, _global_xmax(by_cond))
    axes[0].set_ylim(0, max(10.0, y_max * 1.05))
    _show_all_y_tick_labels(axes)
    fig.suptitle(
        f"Throughput (tasks / min) rolling mean +/- 1 std -- {method_label} baseline\n"
        "window = 10% of total sim time; shared axes across density panels",
        fontsize=12,
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# Plot bounds for the computation-time figure. Both rows are log-scaled
# with shared x and y axes across density panels per the routing team's
# preference (constant axes -> easy cross-density comparison). The TA
# row is capped at 10 s for *plot rendering only*: one outlier TA call
# at 90% / 25 bots took ~2050 s and would otherwise crush the ~1.3 s
# normal range to a single pixel sliver. The cap is annotated in the
# figure title; raw values are still used for the legend's stats.
_TA_PLOT_FLOOR = 1.0
_TA_PLOT_CEIL = 10.0
_TA_YLIM = (1.0, 10.0)
# PF runtime floor is set to 10 ms because (a) sub-10 ms variation is
# below the 'interesting' threshold for path-planning cost, and (b) at
# low robot counts most timesteps have effectively zero PF work, which
# without a sensible floor produces a noisy carpet of vertical drops
# on the log axis. Samples at exactly zero are converted to NaN so the
# line draws a gap rather than dropping to a fake floor.
_PF_PLOT_FLOOR = 1e-2
_PF_YLIM = (1e-2, 1e1)


def _stats_label(robots: int, arr: np.ndarray) -> str:
    """Legend label that doubles as the avg/median/std stats summary."""
    if arr.size == 0:
        return f"{robots} bots"
    return (
        f"{robots} bots  (avg={float(arr.mean()):.2f}, "
        f"med={float(np.median(arr)):.2f}, std={float(arr.std()):.2f})"
    )


def plot_computation_time(by_cond: dict, out_path: Path, method_label: str) -> None:
    """Per-timestep task-allocation and path-planning runtimes (log y).

    Two rows (TA, PF) x three density panels with shared x axis (sim
    timestep) and shared y axis *within each row*. Both rows use a log y
    scale so the wide dynamic range across densities (TA: 1.3 s normal
    vs 2050 s outlier; PF: 0.01 s normal vs 7.6 s peaks) is visible
    without per-panel rescaling. Avg / median / std (computed on raw
    data) are pushed to the legend label rather than into in-panel text
    annotations -- the in-panel annotations were the main source of
    visual clutter on the previous version of this figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey="row")
    row_specs = [
        (
            "Task Allocation Runtime",
            "Task Allocation runtime (s / timestep, log)",
            _TA_PLOT_FLOOR,
            _TA_PLOT_CEIL,
            _TA_YLIM,
        ),
        (
            "Path Planning Runtimes",
            "Path Planning runtime (s / timestep, log)",
            _PF_PLOT_FLOOR,
            None,
            _PF_YLIM,
        ),
    ]

    for row_idx, (json_key, ylabel, plot_floor, plot_ceil, ylim) in enumerate(row_specs):
        for col_idx, density in enumerate(DENSITY_ORDER):
            ax = axes[row_idx, col_idx]
            for robots in sorted(ROBOT_PALETTE):
                run = by_cond.get((density, robots))
                if not run:
                    continue
                series = run.get(json_key) or []
                if not series:
                    continue
                arr = np.asarray(series, dtype=float)
                # Floor (and optionally cap) for log rendering only. Stats
                # in the legend always use the raw ``arr``. Samples that
                # are exactly zero become NaN so the line draws a gap
                # there rather than collapsing to the floor -- this is
                # what cleans up the 10-bot PF panel where most ticks
                # legitimately have no path-planning work to do.
                plot_arr = np.where(arr > 0, np.maximum(arr, plot_floor), np.nan)
                if plot_ceil is not None:
                    plot_arr = np.minimum(plot_arr, plot_ceil)
                t = np.arange(arr.size)
                ax.plot(
                    t,
                    plot_arr,
                    label=_stats_label(robots, arr),
                    color=ROBOT_PALETTE[robots],
                    lw=1.0,
                    alpha=0.55,
                )
            if row_idx == 0:
                ax.set_title(_ax_title_density(density))
            if row_idx == 1:
                ax.set_xlabel("Simulation timestep")
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            ax.set_yscale("log")

        # Apply shared y limits per row (sharey='row' propagates from [0]).
        axes[row_idx, 0].set_ylim(*ylim)
        # Each row gets its own legend (since the stats labels differ per
        # row); place it under the first panel of that row to keep panels
        # uncluttered.
        axes[row_idx, 0].legend(
            loc="upper left",
            fontsize=7.5,
            title=ylabel.split(" runtime")[0] + " stats",
            title_fontsize=8,
        )

    axes[0, 0].set_xlim(0, _global_xmax(by_cond))
    _show_all_y_tick_labels(axes)

    fig.suptitle(
        f"Per-timestep computation time -- {method_label} baseline\n"
        "shared x and y axes across density panels (log y); "
        "TA panel rendered with a 10 s ceiling -- one TA call at 90% / 25 bots "
        "actually peaked at ~2050 s (LNS-PBS internal hang)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _per_task_tardiness(run: dict) -> dict[int, float]:
    """Return ``{task_id: tardiness_seconds}`` for tasks that finished tardy.

    Per-task tardiness = ``max(0, completion_t - deadline_t)``. Reads
    deadlines from ``completed_task_details`` (index 2 of each entry =
    absolute deadline timestep) and completion times from
    ``task_completion_timestamps``. Tasks where either field is missing
    are skipped silently.
    """
    details = run.get("completed_task_details") or {}
    completions = run.get("task_completion_timestamps") or {}
    out: dict[int, float] = {}
    for tid_s, det in details.items():
        try:
            deadline = float(det[2])
            tid_int = int(tid_s)
        except (TypeError, ValueError, IndexError):
            continue
        ct = completions.get(tid_s)
        if ct is None:
            continue
        try:
            ct_f = float(ct)
        except (TypeError, ValueError):
            continue
        tardiness = max(0.0, ct_f - deadline)
        if tardiness > 0:
            out[tid_int] = tardiness
    return out


def plot_cumulative_tardiness(by_cond: dict, out_path: Path, method_label: str) -> None:
    """Cumulative task tardiness (seconds) over simulation time.

    Shared x and y axes across density panels for unambiguous
    cross-density comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    y_max = 0.0
    for ax, density in zip(axes, DENSITY_ORDER):
        for robots in sorted(ROBOT_PALETTE):
            run = by_cond.get((density, robots))
            if not run:
                continue
            n_ticks = int(run.get("timesteps_completed") or 0)
            if n_ticks <= 0:
                continue
            tardiness_by_task = _per_task_tardiness(run)
            completions = run.get("task_completion_timestamps") or {}
            tardy_at_t = np.zeros(n_ticks, dtype=float)
            for tid, t_excess in tardiness_by_task.items():
                ct = completions.get(str(tid))
                if ct is None:
                    continue
                t_complete = int(ct)
                if 0 <= t_complete < n_ticks:
                    tardy_at_t[t_complete] += t_excess
            cumulative = np.cumsum(tardy_at_t)
            t = np.arange(n_ticks)
            ax.plot(t, cumulative, label=f"{robots} bots", color=ROBOT_PALETTE[robots], lw=1.6)
            y_max = max(y_max, float(cumulative[-1]) if cumulative.size else 0.0)
        ax.set_title(_ax_title_density(density))
        ax.set_xlabel("Simulation timestep")
        if density == DENSITY_ORDER[0]:
            ax.set_ylabel("Cumulative task tardiness (s)")
        ax.legend(loc="upper left")
    axes[0].set_xlim(0, _global_xmax(by_cond))
    axes[0].set_ylim(0, max(100.0, y_max * 1.05))
    _show_all_y_tick_labels(axes)
    fig.suptitle(
        f"Cumulative task tardiness -- {method_label} baseline\n"
        "per-task tardiness = max(0, completion_t - deadline_t), summed over completed tasks",
        fontsize=12,
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_tardy_tasks(by_cond: dict, out_path: Path, method_label: str) -> None:
    """Cumulative count of tardy task completions over simulation time.

    Shared x and y axes across density panels.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    y_max = 0
    for ax, density in zip(axes, DENSITY_ORDER):
        for robots in sorted(ROBOT_PALETTE):
            run = by_cond.get((density, robots))
            if not run:
                continue
            n_ticks = int(run.get("timesteps_completed") or 0)
            if n_ticks <= 0:
                continue
            tardiness_by_task = _per_task_tardiness(run)
            completions = run.get("task_completion_timestamps") or {}
            tardy_count_at_t = np.zeros(n_ticks, dtype=int)
            for tid in tardiness_by_task:
                ct = completions.get(str(tid))
                if ct is None:
                    continue
                t_complete = int(ct)
                if 0 <= t_complete < n_ticks:
                    tardy_count_at_t[t_complete] += 1
            cumulative = np.cumsum(tardy_count_at_t)
            t = np.arange(n_ticks)
            ax.plot(t, cumulative, label=f"{robots} bots", color=ROBOT_PALETTE[robots], lw=1.6)
            if cumulative.size:
                y_max = max(y_max, int(cumulative[-1]))
        ax.set_title(_ax_title_density(density))
        ax.set_xlabel("Simulation timestep")
        if density == DENSITY_ORDER[0]:
            ax.set_ylabel("Cumulative tardy tasks")
        ax.legend(loc="upper left")
    axes[0].set_xlim(0, _global_xmax(by_cond))
    axes[0].set_ylim(0, max(5, int(y_max * 1.10) + 1))
    _show_all_y_tick_labels(axes)
    fig.suptitle(
        f"Cumulative tardy tasks -- {method_label} baseline\n"
        "tasks whose completion timestep exceeds their deadline",
        fontsize=12,
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _utilization_series(run: dict, window: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of bots in a productive state (status != 0) per timestep.

    M2M's agent status encoding (per inspection of the run JSONs):
        0 = idle / unallocated, 1 = en-route to pickup, 2 = carrying / dropoff.
    'Productive' here means status > 0; this is the simplest available proxy
    until plan section 3.6's refined utilization (real-task vs rearrangement
    vs idle) is implemented.
    """
    statuses = run.get("agent_statuses_per_timestep") or []
    if not statuses:
        return np.array([]), np.array([])
    arr = np.asarray(statuses)
    util = (arr > 0).mean(axis=1)
    if window > 1 and util.size >= window:
        kernel = np.ones(window) / window
        util_smooth = np.convolve(util, kernel, mode="valid")
        offset = (window - 1) // 2
        t = np.arange(offset, offset + util_smooth.size)
        return t, util_smooth
    return np.arange(util.size), util


def plot_bot_utilization(by_cond: dict, out_path: Path, method_label: str) -> None:
    """Productive bot fraction with shared x and y axes."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for ax, density in zip(axes, DENSITY_ORDER):
        for robots in sorted(ROBOT_PALETTE):
            run = by_cond.get((density, robots))
            if not run:
                continue
            t, util = _utilization_series(run)
            if util.size == 0:
                continue
            ax.plot(
                t,
                util,
                label=f"{robots} bots",
                color=ROBOT_PALETTE[robots],
                lw=1.6,
            )
        ax.set_title(_ax_title_density(density))
        ax.set_xlabel("Simulation timestep")
        if density == DENSITY_ORDER[0]:
            ax.set_ylabel("Productive bot fraction")
        ax.legend(loc="lower right")
    axes[0].set_xlim(0, _global_xmax(by_cond))
    axes[0].set_ylim(0, 1.02)
    _show_all_y_tick_labels(axes)
    fig.suptitle(
        f"Bot utilization (status > 0, 11-step rolling mean) -- {method_label} baseline",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str]) -> int:
    raw = Path(argv[1]) if len(argv) > 1 else Path("data/raw_data")
    out = Path(argv[2]) if len(argv) > 2 else Path("data/figures")
    out.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(raw)
    if not runs:
        print(f"No runs found in {raw}", file=sys.stderr)
        return 1

    by_cond = _index_by_condition(runs)
    method_label = _detect_method_label(runs)
    print(
        f"Loaded {len(runs)} runs, {len(by_cond)} (density, robots) conditions; "
        f"method = {method_label}."
    )

    _set_paper_style()

    plot_throughput_rolling(by_cond, out / "throughput_rolling.png", method_label)
    plot_sku_spread(by_cond, out / "sku_spread_trajectories.png", method_label)
    plot_bot_utilization(by_cond, out / "bot_utilization_trajectories.png", method_label)
    plot_computation_time(by_cond, out / "computation_time_trajectories.png", method_label)
    plot_cumulative_tardiness(by_cond, out / "cumulative_tardiness.png", method_label)
    plot_cumulative_tardy_tasks(by_cond, out / "cumulative_tardy_tasks.png", method_label)

    # Clean up superseded figures so stale outputs aren't accidentally
    # presented alongside the current set.
    for stale in ("throughput_heatmap.png", "cumulative_completed_tasks.png"):
        stale_path = out / stale
        if stale_path.exists():
            stale_path.unlink()

    print("Wrote:")
    for name in (
        "throughput_rolling.png",
        "sku_spread_trajectories.png",
        "bot_utilization_trajectories.png",
        "computation_time_trajectories.png",
        "cumulative_tardiness.png",
        "cumulative_tardy_tasks.png",
    ):
        print(f"  {out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
