"""Overlay real-task throughput for crM2M against the three baselines
(M2M, LNS-PBS, HBH+MLA*) on a single panel.

This is the same throughput definition and styling as
``plot_crm2m_compare.plot_throughput_rolling`` (per-minute binned tasks/min,
centered rolling mean +/- 1 std band, team paper style) but overlays four
conditions instead of two and writes to its own filename so the six
``*_crm2m-vs-m2m.png`` comparison figures are left untouched.

All four runs must share the same config (1-hour / 3600-tick, 40 bots, 30%
inventory, queue, deadlines off) so the only difference is the allocation
method. Throughput counts completed *real* tasks only -- shuffles never count.

Usage:
    python scripts/plot_throughput_4way.py M2M.json CRM2M.json LNSPBS.json HBH.json [out.png]
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
    ("crM2M", "#d62728"),     # red
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
    # Throughput sits high (~100-140/min) so the upper-right collides with the
    # lines; the lower-right corner is empty -> park the legend there.
    ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Throughput (tasks / min) rolling mean +/- 1 std -- crM2M vs baselines\n"
        "1-minute bins; rolling window = min(5 min, n_bins/3); std = minute-to-minute jitter",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 4:
        sys.exit(
            "usage: plot_throughput_4way.py M2M.json CRM2M.json LNSPBS.json HBH.json [out.png]"
        )
    paths = args[:4]
    out = Path(args[4] if len(args) > 4 else "data/figures/throughput_rolling_4way-baselines.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    pb._set_paper_style()
    runs = [json.load(open(p)) for p in paths]
    plot_throughput_rolling(runs, SERIES, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
