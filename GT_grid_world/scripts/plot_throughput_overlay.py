"""Overlay real-task throughput for a crM2M lambda comparison, in the team's
``plot_crm2m_compare`` style (single panel, per-minute binned tasks/min with a
centered rolling mean +/- 1 std band).

Mirrors ``plot_crm2m_compare.plot_throughput_rolling`` exactly (same figsize,
binning, rolling-window rule, legend placement, paper style) but overlays two
lambda conditions instead of M2M vs crM2M, and writes to its own filename so the
existing ``*_crm2m-vs-m2m.png`` figures are left untouched.

Usage:
    python scripts/plot_throughput_overlay.py [out.png]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_baselines as pb

DEFAULT_WINDOW_MINUTES = 5

# One entry per overlaid condition: (json path, legend label, colour).
RUNS = [
    ("data/raw_data/RESULT_lambda1e9_40bot_30pct_30min_900.json", r"$\lambda = 10^9$", "#1f77b4"),
    ("data/raw_data/RESULT_lambda1p5_cut10_40bot_30pct_30min_900.json", r"$\lambda = 1.5$", "#d62728"),
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
        ax.fill_between(t, lower, mean + std, color=color, alpha=0.18, linewidth=0)
        ax.plot(t, mean, label=label, color=color, lw=1.8)
        y_max = max(y_max, float((mean + std).max()) if mean.size else 0.0)
    ax.set_xlim(0, _xmax(runs))
    ax.set_ylim(0, max(10.0, y_max * 1.05))
    ax.set_xlabel("Simulation timestep")
    ax.set_ylabel("Throughput (tasks / min)")
    ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "Throughput (tasks / min) rolling mean +/- 1 std -- crM2M lambda sweep\n"
        "1-minute bins; rolling window = min(5 min, n_bins/3); std = minute-to-minute jitter",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/figures/throughput_rolling_crm2m-lambda-sweep.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    pb._set_paper_style()
    runs = [json.load(open(p)) for p, _, _ in RUNS]
    labels = [(lbl, color) for _, lbl, color in RUNS]
    plot_throughput_rolling(runs, labels, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
