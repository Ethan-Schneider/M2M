"""Summarise headline metrics across the M2M baseline sweep.

Reads every run JSON in ``data/raw_data/`` (or a directory passed as an
argument), filters to runs matching the M2M-baseline sweep filename pattern
written by ``run_baselines.sh``, and prints a compact table of the headline
metrics relevant for roadmap section 1.7:

    * timesteps_completed (= ticks before the wall-clock budget hit)
    * total_completed_tasks
    * throughput (tasks/min)
    * average_service_time
    * overdue_task_completions
    * SKU Spread: final, mean, and delta vs initial
    * Total Runtime (wall-clock seconds)

Usage:
    python scripts/summarize_baselines.py [data/raw_data]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean


def _fmt(x, fmt: str = "{:.2f}") -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return fmt.format(x)
    return str(x)


def summarise(path: Path) -> dict | None:
    try:
        with path.open() as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    spread = d.get("sku_spread_per_timestep") or []
    spread_first = spread[0] if spread else None
    spread_last = spread[-1] if spread else None
    spread_mean = mean(spread) if spread else None
    spread_delta = (
        spread_last - spread_first
        if spread_first is not None and spread_last is not None
        else None
    )

    return {
        "name": path.name,
        "robots": d.get("num_robots"),
        "density": d.get("initial_inventory"),
        "ticks": d.get("timesteps_completed"),
        "completed": d.get("total_completed_tasks"),
        "throughput": d.get("throughput (tasks/min)"),
        "avg_service": d.get("average_service_time"),
        "overdue": d.get("overdue_task_completions"),
        "spread_first": spread_first,
        "spread_last": spread_last,
        "spread_mean": spread_mean,
        "spread_delta": spread_delta,
        "runtime_s": d.get("Total Runtime"),
    }


METHOD_LABELS = {
    "c_lns": "LNS-PBS",
    "py_lns": "M2M (4D cost tensor + py_lns)",
    "hbh_mla_star": "HBH+MLA*",
}


def _detect_method_label(files: list[Path]) -> str:
    strategies: set[str] = set()
    for p in files:
        try:
            with p.open() as f:
                d = json.load(f)
            s = d.get("improvement_task_assignment_strategy")
            if s:
                strategies.add(s)
        except (OSError, json.JSONDecodeError):
            continue
    if not strategies:
        return "unknown method"
    if len(strategies) == 1:
        s = next(iter(strategies))
        return METHOD_LABELS.get(s, s)
    return "mixed (" + ", ".join(sorted(METHOD_LABELS.get(s, s) for s in strategies)) + ")"


def _filter_by_strategy(files: list[Path], strategy: str) -> list[Path]:
    """Keep only the JSONs whose ``improvement_task_assignment_strategy``
    matches ``strategy``. Used by ``--method`` so the summary table can
    isolate one allocator at a time when multiple are present in the
    same ``raw_data/`` directory.
    """
    out: list[Path] = []
    for p in files:
        try:
            with p.open() as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("improvement_task_assignment_strategy") == strategy:
            out.append(p)
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "root", nargs="?", default="data/raw_data",
        help="Directory of run JSONs to load (default: data/raw_data).",
    )
    parser.add_argument(
        "--method",
        choices=["all", "c_lns", "py_lns", "hbh_mla_star"],
        default="all",
        help=(
            "Restrict the summary to a single allocator's runs (matched on "
            "the JSON's improvement_task_assignment_strategy field). "
            "Default 'all' matches the historical behaviour of the script."
        ),
    )
    args = parser.parse_args(argv[1:])

    root = Path(args.root)
    files = sorted(root.glob("*.json"))
    if not files:
        print(f"No JSON outputs found in {root}", file=sys.stderr)
        return 1
    if args.method != "all":
        files = _filter_by_strategy(files, args.method)
        if not files:
            print(
                f"No runs in {root} matched --method={args.method!r}",
                file=sys.stderr,
            )
            return 1

    method_label = _detect_method_label(files)
    print(f"Method: {method_label}")
    print()

    rows = [r for r in (summarise(p) for p in files) if r is not None]

    rows.sort(
        key=lambda r: (
            r["density"] if r["density"] is not None else -1,
            r["robots"] if r["robots"] is not None else -1,
        )
    )

    header = (
        f"{'density':>7} {'robots':>6} {'ticks':>6} {'done':>5} "
        f"{'tput/min':>9} {'avg_svc':>8} {'overdue':>7} "
        f"{'EZC0':>7} {'EZCf':>7} {'dEZC':>7} {'EZCmean':>8} "
        f"{'wall(s)':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{_fmt(r['density'], '{:.0f}'):>7} "
            f"{_fmt(r['robots']):>6} "
            f"{_fmt(r['ticks']):>6} "
            f"{_fmt(r['completed']):>5} "
            f"{_fmt(r['throughput']):>9} "
            f"{_fmt(r['avg_service']):>8} "
            f"{_fmt(r['overdue']):>7} "
            f"{_fmt(r['spread_first'], '{:.1f}'):>7} "
            f"{_fmt(r['spread_last'], '{:.1f}'):>7} "
            f"{_fmt(r['spread_delta'], '{:+.1f}'):>7} "
            f"{_fmt(r['spread_mean'], '{:.1f}'):>8} "
            f"{_fmt(r['runtime_s'], '{:.0f}'):>7}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
