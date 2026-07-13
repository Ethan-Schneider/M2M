#!/usr/bin/env python3
"""Regenerate Step-A bottleneck report figures from run JSONs.

Writes kpi_comparison.png and bottleneck_decomposition.png for a controlled
M2M vs crM2M pair (typically the m=0 baseline on the corrected sim).

Usage:
    python build_bottleneck_figures.py M2M.json crM2M.json --out-dir REPORT_DIR
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def _metrics(path: Path) -> dict:
    d = json.load(path.open())
    H = int(d["timesteps_completed"])
    K = int(d["buffer_capacity_k"])
    N = int(d["num_robots"])
    done = int(d["total_completed_tasks"])
    buf = np.array([float(v) for v in d["output_buffer_level_per_timestep"].values()])
    stat = np.array(d["stationary_robots"][:H], float)
    s2p = [float(v) for v in d["actual_distance_start_to_pick"].values()]
    p2p = [float(v) for v in d["actual_distance_of_task_from_pick_to_place"].values()]
    return {
        "thr": done / (H / 60),
        "done": done,
        "shuf": int(d.get("total_completed_rearrangement_tasks") or 0),
        "bufutil": float(buf.mean() / K * 100),
        "wait": float(stat.mean() / N * 100),
        "blocks": int(d.get("outbound_buffer_placement_blocks") or 0),
        "retr": float(np.mean(s2p)) if s2p else float("nan"),
        "fulfil": float(np.mean(p2p)) if p2p else float("nan"),
        "pf": int(d.get("Number of Path Plan Fails") or 0),
    }


def plot_kpi(m: dict, c: dict, out_path: Path, *, title_suffix: str) -> None:
    panels = [
        ("Throughput (tasks/min)", "thr", "{:.2f}"),
        ("Buffer utilization (%)", "bufutil", "{:.1f}"),
        ("Bot wait / stationary (%)", "wait", "{:.1f}"),
        ("Buffer-full blocks", "blocks", "{:,}"),
        ("Retrieval travel (start->pick)", "retr", "{:.2f}"),
        ("Fulfilment travel (pick->place)", "fulfil", "{:.2f}"),
        ("PBS path-plan fails", "pf", "{:,}"),
        ("Rearrangements completed", "shuf", "{:d}"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for ax, (ptitle, key, fmt) in zip(axes.ravel(), panels):
        vals = [m[key], c[key]]
        bars = ax.bar(["M2M", "crM2M"], vals, color=["tab:blue", "tab:orange"], width=0.6)
        ax.set_title(ptitle, fontsize=11, fontweight="bold")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=10)
        ax.margins(y=0.18)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"Bottleneck diagnosis: M2M vs crM2M ({title_suffix})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("m2m", type=Path)
    ap.add_argument("crm2m", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--title-suffix",
        default="30 min, K=20, mu=25, N=40, m=0, corrected sim",
    )
    args = ap.parse_args()

    m = _metrics(args.m2m)
    c = _metrics(args.crm2m)
    out = args.out_dir
    plot_kpi(m, c, out / "kpi_comparison.png", title_suffix=args.title_suffix)

    decomp = out / "bottleneck_decomposition.png"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "bottleneck_analysis.py"),
            f"M2M={args.m2m}",
            f"crM2M={args.crm2m}",
            "--out", str(decomp),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
