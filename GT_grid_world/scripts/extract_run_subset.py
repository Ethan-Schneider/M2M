"""Extract a slim subset of GT_grid_world run JSON into smaller copies.

Edit ``INPUT_FILES`` below with the raw-data JSON paths you want to slim,
then run:

    python scripts/extract_run_subset.py
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Fill in the list of full run JSON files to slim.
# Paths may be absolute or relative to the M2M repo root.
# ---------------------------------------------------------------------------
INPUT_FILES: list[str] = [
    # "data/raw_data/3600_precomputed_queue_....json",
    # "./data/raw_data/3600_precomputed_queue_simultaneous_4.0_small_restricted_longer_adversarial_60_40.0_fast_greedy_M2M_pbs_study_small_restricted_longer_40_120_1.0_0.25_0.0_none_none_1300.json",
    # "./data/raw_data/3600_precomputed_queue_simultaneous_4.0_small_restricted_longer_adversarial_60_40.0_fast_greedy_M2M_pbs_study_small_restricted_longer_40_120_1.0_0.25_0.0_none_none_1301.json",
    # "./data/raw_data/3600_precomputed_queue_simultaneous_4.0_small_restricted_longer_adversarial_60_40.0_fast_greedy_M2M_pbs_study_small_restricted_longer_40_120_1.0_0.25_0.0_none_none_1350.json",
    # "./data/raw_data/3600_precomputed_queue_simultaneous_4.0_small_restricted_longer_adversarial_60_40.0_fast_greedy_M2M_pbs_study_small_restricted_longer_40_120_1.0_0.25_0.0_none_none_1351.json",
    # "./data/raw_data/3600_precomputed_queue_simultaneous_4.0_small_restricted_longer_adversarial_60_40.0_fast_greedy_M2M_pbs_study_small_restricted_longer_40_120_1.0_0.25_0.0_none_none_1352.json",
    # "./data/raw_data/3600_precomputed_queue_simultaneous_4.0_small_restricted_longer_adversarial_60_40.0_fast_greedy_M2M_pbs_study_small_restricted_longer_40_120_1.0_0.25_0.0_none_none_1310.json",
    # "./data/raw_data/3600_precomputed_queue_simultaneous_4.0_small_restricted_longer_adversarial_60_40.0_fast_greedy_M2M_pbs_study_small_restricted_longer_40_120_1.0_0.25_0.0_none_none_1311.json",
    # "./data/raw_data/3600_precomputed_queue_simultaneous_4.0_small_restricted_longer_adversarial_60_40.0_fast_greedy_M2M_pbs_study_small_restricted_longer_40_120_1.0_0.25_0.0_none_none_1312.json",
    # "./data/raw_data/3600_precomputed_queue_simultaneous_4.0_small_restricted_longer_adversarial_60_40.0_fast_greedy_M2M_pbs_study_small_restricted_longer_40_120_1.0_0.25_0.0_none_none_1320.json",
    # "./data/raw_data/3600_precomputed_queue_simultaneous_4.0_small_restricted_longer_adversarial_60_40.0_fast_greedy_M2M_pbs_study_small_restricted_longer_40_120_1.0_0.25_0.0_none_none_1321.json",
    # "./data/raw_data/3600_precomputed_queue_simultaneous_4.0_small_restricted_longer_adversarial_60_40.0_fast_greedy_M2M_pbs_study_small_restricted_longer_40_120_1.0_0.25_0.0_none_none_1322.json",
    './data/raw_data/3600_precomputed_queue_simultaneous_1.0_small_restricted_longer_adversarial_60_40.0_fast_greedy_M2M_pbs_study_small_restricted_longer_40_120_1.0_0.0_0.0_none_none_1450.json',
    './data/raw_data/3600_precomputed_queue_simultaneous_1.0_small_restricted_longer_adversarial_60_40.0_fast_greedy_M2M_pbs_study_small_restricted_longer_40_120_1.0_0.0_0.0_none_none_1451.json',
    './data/raw_data/3600_precomputed_queue_simultaneous_1.0_small_restricted_longer_adversarial_60_40.0_fast_greedy_M2M_pbs_study_small_restricted_longer_40_120_1.0_0.0_0.0_none_none_1452.json',
]

# Written next to each input as ``<stem>_subset.json`` unless overridden.
OUTPUT_SUFFIX = "_subset"
OUTPUT_DIR: str | None = None  # e.g. "data/raw_data/subsets"; None = same dir as input

# Config / hyperparameter keys (statistics.save_data input block).
CONFIG_KEYS = [
    "seed",
    "num_robots",
    "time_horizon",
    "max_tasks",
    "task_generation_strategy",
    "initial_task_assignment_strategy",
    "improvement_task_assignment_strategy",
    "path_planning_strategy",
    "map_name",
    "time_limit",
    "visualize_output",
    "initial_inventory",
    "frequency",
    "inbound_outbound_ratio",
    "output_graphs",
    "num_skus",
    "weight_init_method",
    "cost_calculation_method",
    "removal_operator",
    "repair_operator",
    "acceptance_function",
    "T_0",
    "alpha",
    "deadline_generation_method",
    "deadline_offset",
    "output_intermediate_data",
    "intermediate_data_interval",
    "base_cost_weight",
    "deadline_weight",
    "sku_distribution_weight",
    "agent_unallocated_penalty",
    "solution_repair_detection_function",
    "solution_repair_function",
    "schedule_name",
    "W",
    "B",
    "lambda_",
    "reallocation_task_method",
    "queue_release_window",
    "pick_place_time",
    "buffer_capacity_k",
    "buffer_consumption_rate",
    "use_precomputed_queue",
    "queue_file",
    "initial_inventory_file",
    "run_until_queue_complete",
    "log_buffer_predictions",
]

# Simulation result keys requested for the slim copy.
RESULT_KEYS = [
    "agent_task_per_timestep",
    "task_completion_timestamps",
    "throughput (tasks/min)",
    "actual_duration_of_task_from_start_to_pick",
    "actual_duration_of_task_from_pick_to_place",
]

KEEP_KEYS = CONFIG_KEYS + RESULT_KEYS


def _repo_root() -> Path:
    # scripts/ -> GT_grid_world/ -> M2M/
    return Path(__file__).resolve().parents[2]


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = _repo_root() / p
    return p


def _output_path(input_path: Path) -> Path:
    out_name = f"{input_path.stem}{OUTPUT_SUFFIX}.json"
    if OUTPUT_DIR is None:
        return input_path.with_name(out_name)
    out_dir = _resolve(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / out_name


def extract_subset(data: dict) -> dict:
    subset = {}
    missing = []
    for key in KEEP_KEYS:
        if key in data:
            subset[key] = data[key]
        else:
            missing.append(key)
    return subset, missing


def main() -> None:
    if not INPUT_FILES:
        raise SystemExit(
            "INPUT_FILES is empty. Add JSON paths at the top of this script."
        )

    for raw in INPUT_FILES:
        src = _resolve(raw)
        if not src.is_file():
            print(f"[skip] not found: {src}")
            continue

        with src.open() as f:
            data = json.load(f)

        subset, missing = extract_subset(data)
        dst = _output_path(src)
        with dst.open("w") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)

        print(f"[ok] {src.name} -> {dst} ({len(subset)} keys)")
        if missing:
            print(f"     missing keys: {missing}")


if __name__ == "__main__":
    main()
