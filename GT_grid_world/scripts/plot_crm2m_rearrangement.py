"""crM2M-only rearrangement diagnostics: benefit, detour cost, and utility of
*completed* shuffle tasks.

Baselines (M2M, HBH+MLA*, LNS-PBS) never rearrange, so these figures are only
meaningful for a crM2M run and are generated for it alone. For every completed
rearrangement task the simulator records the objective terms the allocator
scored it with (see ``committed_shuffle_terms`` / ``record_rearrangement_task_score``):

    benefit b        = dist(s_p, h0) - dist(d_q, h0)     (placement improvement)
    detour  Delta    = dist(prev, s_p) + dist(s_p, h0) - dist(prev, h0)
    utility U        = b - lambda * max(0, (Delta + c_pp) - slack)

Two figures are produced:

  1. rearrangement_benefit_cost -- per completed shuffle, benefit vs detour cost
     over time (scatter + rolling mean), with cumulative totals.
  2. rearrangement_utility      -- per completed shuffle, realized utility over
     time (scatter + rolling mean), with the U>0 reference line.

Usage:
    python scripts/plot_crm2m_rearrangement.py CRM2M.json [OUT_DIR] [--tag TAG]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_baselines as pb

BENEFIT_COLOR = "#2ca02c"
COST_COLOR = "#ff7f0e"
UTILITY_COLOR = "#d62728"
CRM2M_COLOR = "#d62728"


def _load(fp: Path) -> dict:
    with fp.open() as f:
        return json.load(f)


def _completed_rearrangement_series(run: dict):
    """Join per-shuffle objective scores with completion timestamps.

    Returns ``(t, benefit, detour, utility)`` numpy arrays sorted by completion
    timestep, restricted to shuffles that both completed and were scored.
    """
    scores = run.get("rearrangement_task_scores") or {}
    completions = run.get("rearrangement_task_completion_timestamps") or {}
    rows = []
    for tid, ct in completions.items():
        score = scores.get(str(tid))
        if score is None:
            continue
        rows.append(
            (int(ct), float(score["benefit"]), float(score["detour"]),
             float(score["utility"]))
        )
    if not rows:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty
    rows.sort(key=lambda r: r[0])
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def _rolling(y: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean; falls back to the raw series when too short."""
    if y.size == 0 or window <= 1 or y.size < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def _xmax(run: dict, t: np.ndarray) -> int:
    n = int(run.get("timesteps_completed") or 0)
    if n > 0:
        return n
    return int(t.max()) + 1 if t.size else 500


def _empty_panel(ax, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="#555555")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_benefit_cost(run, out_path, tag="crm2m") -> None:
    t, benefit, detour, _ = _completed_rearrangement_series(run)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if t.size == 0:
        _empty_panel(ax, "No completed rearrangement tasks recorded\n"
                          "(0 shuffles committed / completed)")
        fig.suptitle("Rearrangement benefit vs detour cost -- crM2M", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    window = max(1, min(25, t.size // 5))
    ax.scatter(t, benefit, s=14, color=BENEFIT_COLOR, alpha=0.35,
               label="benefit b (per task)")
    ax.scatter(t, detour, s=14, color=COST_COLOR, alpha=0.35,
               label="detour cost Delta (per task)")
    if t.size >= window > 1:
        ax.plot(t, _rolling(benefit, window), color=BENEFIT_COLOR, lw=2.0,
                label=f"benefit (rolling {window})")
        ax.plot(t, _rolling(detour, window), color=COST_COLOR, lw=2.0,
                label=f"detour (rolling {window})")
    ax.set_xlim(0, _xmax(run, t))
    ax.set_xlabel("Completion timestep")
    ax.set_ylabel("Cells (shortest-path distance)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    stats = (f"N completed = {t.size}\n"
             f"mean benefit = {benefit.mean():.2f}\n"
             f"mean detour  = {detour.mean():.2f}\n"
             f"total benefit = {benefit.sum():.0f}\n"
             f"total detour  = {detour.sum():.0f}")
    ax.text(0.01, 0.98, stats, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.85))
    fig.suptitle(
        "Rearrangement benefit vs detour cost per completed shuffle -- crM2M\n"
        "benefit = aisle-exit improvement; detour = marginal travel cost",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_utility(run, out_path, tag="crm2m") -> None:
    t, _, _, utility = _completed_rearrangement_series(run)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if t.size == 0:
        _empty_panel(ax, "No completed rearrangement tasks recorded\n"
                          "(0 shuffles committed / completed)")
        fig.suptitle("Rearrangement utility -- crM2M", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    window = max(1, min(25, t.size // 5))
    ax.axhline(0.0, color="#888888", lw=1.0, ls="--", label="U = 0")
    ax.scatter(t, utility, s=16, color=UTILITY_COLOR, alpha=0.4,
               label="utility U (per task)")
    if t.size >= window > 1:
        ax.plot(t, _rolling(utility, window), color=UTILITY_COLOR, lw=2.0,
                label=f"utility (rolling {window})")
    ax.set_xlim(0, _xmax(run, t))
    ax.set_xlabel("Completion timestep")
    ax.set_ylabel("Realized utility U")
    ax.legend(loc="upper right", fontsize=8)
    stats = (f"N completed = {t.size}\n"
             f"mean U   = {utility.mean():.2f}\n"
             f"median U = {np.median(utility):.2f}\n"
             f"min / max = {utility.min():.2f} / {utility.max():.2f}\n"
             f"total U  = {utility.sum():.1f}")
    ax.text(0.01, 0.98, stats, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.85))
    fig.suptitle(
        "Realized utility per completed shuffle -- crM2M\n"
        "U = benefit - lambda * max(0, (Delta + c_pp) - slack)",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("crm2m", type=Path)
    ap.add_argument("out_dir", type=Path, nargs="?", default=Path("data/figures"))
    ap.add_argument("--tag", default="crm2m",
                    help="filename suffix (default: crm2m)")
    args = ap.parse_args()

    pb._set_paper_style()
    run = _load(args.crm2m)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        ("rearrangement_benefit_cost", plot_benefit_cost),
        ("rearrangement_utility", plot_utility),
    ]
    for name, fn in figures:
        out = args.out_dir / f"{name}_{args.tag}.png"
        fn(run, out, args.tag)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
