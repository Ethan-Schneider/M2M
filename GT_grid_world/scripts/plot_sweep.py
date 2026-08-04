#!/usr/bin/env python3
"""Aggregate bottleneck-sweep runs into throughput-vs-parameter figures.

Scans a directory of ``{method}_N{N}_K{K}_mu{MU}_mt{MT}.json`` run outputs
(method in {m2m, crm2m}) and, for each swept axis (mu, K, N, MT), plots M2M vs
crM2M throughput while the other three knobs sit at their baseline
(N=40, K=20, mu=25, MT=120). Also prints a full metrics table.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = {"N": 40, "K": 20, "MU": 25, "MT": 120}
NAME_RE = re.compile(r"(m2m|crm2m)_N(\d+)_K(\d+)_mu(\d+)_mt(\d+)\.json$")


def load_metrics(path):
    d = json.load(open(path))
    H = int(d.get("timesteps_completed") or d.get("time_horizon") or 1800)
    K = int(d.get("buffer_capacity_k") or 0)
    N = int(d.get("num_robots") or 0)
    completed = int(d.get("total_completed_tasks") or 0)
    thr = completed / (H / 60.0) if H else 0.0
    shuffles = int(d.get("total_completed_rearrangement_tasks") or 0)
    buf = d.get("output_buffer_level_per_timestep", {})
    bvals = np.array([float(v) for v in buf.values()]) if isinstance(buf, dict) else np.array(buf, float)
    buf_util = float(bvals.mean() / K) if (K and bvals.size) else float("nan")
    stat = np.array(d.get("stationary_robots", [])[:H], float)
    wait = float(stat.mean() / N) if (N and stat.size) else float("nan")
    blocks = int(d.get("outbound_buffer_placement_blocks") or 0)
    s2p = d.get("actual_distance_start_to_pick", {})
    retr = float(np.mean([float(v) for v in s2p.values()])) if s2p else float("nan")
    return dict(H=H, thr=thr, completed=completed, shuffles=shuffles,
                buf_util=buf_util, wait=wait, blocks=blocks, retr=retr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--out-dir", default="data/figures/bottleneck_sweep")
    args = ap.parse_args()

    runs = {}  # (method, N,K,MU,MT) -> metrics
    for p in sorted(Path(args.sweep_dir).glob("*.json")):
        m = NAME_RE.search(p.name)
        if not m:
            continue
        method, N, K, MU, MT = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        runs[(method, N, K, MU, MT)] = load_metrics(p)

    # ---- table ----
    hdr = f"{'method':7}{'N':>4}{'K':>5}{'mu':>5}{'MT':>5}{'thru/min':>10}{'done':>7}{'shuf':>6}{'bufutil':>9}{'wait':>7}{'blocks':>8}{'retr':>7}"
    print(hdr); print("-" * len(hdr))
    for key in sorted(runs):
        method, N, K, MU, MT = key
        r = runs[key]
        print(f"{method:7}{N:>4}{K:>5}{MU:>5}{MT:>5}{r['thr']:>10.2f}{r['completed']:>7}{r['shuffles']:>6}{r['buf_util']:>9.1%}{r['wait']:>7.1%}{r['blocks']:>8}{r['retr']:>7.2f}")

    # ---- per-axis figures ----
    axes_spec = [
        ("MU", "buffer drain rate mu (tasks/min)", lambda N, K, MU, MT: N == BASE["N"] and K == BASE["K"] and MT == BASE["MT"], "B1_throughput_vs_mu"),
        ("K", "buffer capacity K", lambda N, K, MU, MT: N == BASE["N"] and MU == BASE["MU"] and MT == BASE["MT"], "B1_throughput_vs_K"),
        ("MT", "offered load / WIP cap (max-tasks)", lambda N, K, MU, MT: N == BASE["N"] and K == BASE["K"] and MU == BASE["MU"], "B2_throughput_vs_load"),
        ("N", "number of robots N", lambda N, K, MU, MT: K == BASE["K"] and MU == BASE["MU"] and MT == BASE["MT"], "B3_throughput_vs_bots"),
    ]
    idx = {"N": 1, "K": 2, "MU": 3, "MT": 4}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _key_from_stem(stem):
        m = NAME_RE.search(stem + ".json")
        if not m:
            return None
        return (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))

    for axis, xlabel, keep, fname in axes_spec:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for method, color in (("m2m", "tab:blue"), ("crm2m", "tab:orange")):
            pts = []
            for key, r in runs.items():
                if key[0] != method:
                    continue
                _, N, K, MU, MT = key
                if keep(N, K, MU, MT):
                    pts.append((key[idx[axis]], r["thr"]))
            pts.sort()
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "o-", color=color, label=method, lw=1.8, ms=6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("throughput (tasks/min)")
        ax.set_title(f"Throughput vs {xlabel}\n(other knobs at baseline N=40,K=20,mu=25,MT=120)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fp = out_dir / f"{fname}.png"
        fig.savefig(fp, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"wrote {fp}")

    # ---- 4-panel summary (explicit x-values; do not parse filenames) ----
    summary_panels = [
        ("Drain rate mu (K=20, N=40)", [25, 40, 60, 120],
         lambda v: (f"m2m_N40_K20_mu{v}_mt120", f"crm2m_N40_K20_mu{v}_mt120")),
        ("Buffer capacity K (mu=25, N=40)", [20, 40, 80],
         lambda v: (f"m2m_N40_K{v}_mu25_mt120", f"crm2m_N40_K{v}_mu25_mt120")),
        ("Work-in-progress cap (mu=25, K=20, N=40)", [40, 80, 120],
         lambda v: (f"m2m_N40_K20_mu25_mt{v}", f"crm2m_N40_K20_mu25_mt{v}")),
        ("Fleet size N (mu=25, K=20)", [20, 30, 40, 60],
         lambda v: (f"m2m_N{v}_K20_mu25_mt120", f"crm2m_N{v}_K20_mu25_mt120")),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (title, xs, names_for) in zip(axes.ravel(), summary_panels):
        m2m_y, cr_y, shuf, plot_xs = [], [], [], []
        for x in xs:
            m2m_name, cr_name = names_for(x)
            k1 = _key_from_stem(m2m_name)
            k2 = _key_from_stem(cr_name)
            r1, r2 = runs.get(k1), runs.get(k2)
            if r1 is None or r2 is None:
                print(f"WARN: missing sweep run for panel {title!r} x={x}")
                continue
            plot_xs.append(x)
            m2m_y.append(r1["thr"])
            cr_y.append(r2["thr"])
            shuf.append(r2["shuffles"])
        ax.plot(plot_xs, m2m_y, "o-", color="tab:blue", label="M2M", lw=2, ms=7)
        ax.plot(plot_xs, cr_y, "o-", color="tab:orange", label="crM2M", lw=2, ms=7)
        for x, t, s in zip(plot_xs, cr_y, shuf):
            if s > 0:
                ax.annotate(f"{s} shuf", (x, t), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8, color="tab:orange")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("throughput (tasks/min)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Parameter sweep: crM2M never meaningfully separates from M2M",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    summary_fp = out_dir / "sweep_summary_4panel.png"
    fig.savefig(summary_fp, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"wrote {summary_fp}")


if __name__ == "__main__":
    main()
