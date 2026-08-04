"""Overlay real-task throughput for the fully-ONLINE (no-queue) baseline
comparison: M2M vs LNS-PBS vs HBH+MLA*.

Same throughput definition and styling as
``plot_crm2m_compare.plot_throughput_rolling`` (per-minute binned tasks/min,
centered rolling mean +/- 1 std band, team paper style); writes its own file
so nothing else is clobbered. Colours match ``plot_throughput_4way`` so the
online and queue figures are visually comparable.

All runs must share config (here: 30-min / 1800-tick, 40 bots, 30% inventory,
ONLINE task generation -- no precomputed queue, deadlines off). Throughput
counts completed real tasks only.

Usage:
    python scripts/plot_throughput_3way.py M2M.json LNSPBS.json HBH.json [out.png]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_baselines as pb

DEFAULT_WINDOW_MINUTES = 5

# (legend label, colour) in the positional order the run JSONs are passed.
SERIES = [
    ("M2M", "#1f77b4"),       # blue
    ("LNS-PBS", "#2ca02c"),   # green
    ("HBH+MLA*", "#ff7f0e"),  # orange
]


def _xmax(runs):
    return max((int(r.get("timesteps_completed") or 0) for r in runs), default=500) or 500


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
    # Throughput sits high so the upper-right collides with the lines; the
    # lower-right corner is empty -> park the legend there.
    ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Throughput (tasks / min) rolling mean +/- 1 std -- ONLINE (no queue)\n"
        "1-minute bins; rolling window = min(5 min, n_bins/3); std = minute-to-minute jitter",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 3:
        sys.exit("usage: plot_throughput_3way.py M2M.json LNSPBS.json HBH.json [out.png]")
    paths = args[:3]
    out = Path(args[3] if len(args) > 3 else "data/figures/throughput_rolling_3way-online.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    pb._set_paper_style()
    runs = [json.load(open(p)) for p in paths]
    plot_throughput_rolling(runs, SERIES, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
