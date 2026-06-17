import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from numpy.typing import NDArray

import functions

TASK_TYPE_OUTBOUND = 0
TASK_TYPE_INBOUND = 1

SCHEDULE_RELEASE_TIME = 0
SCHEDULE_DEADLINE = 1
SCHEDULE_SKU_ID = 2
SCHEDULE_TASK_TYPE = 3


def get_deadline(current_time: int, deadline_generation_method: str, deadline_offset: float) -> int:
    """Mirror GT_grid_world case_request_generator.get_deadline."""
    if deadline_generation_method == "constant":
        return current_time + int(deadline_offset)
    if deadline_generation_method == "normal":
        deadline = np.random.normal(loc=deadline_offset, scale=3)
        return current_time + int(round(deadline))
    if deadline_generation_method == "bimodal":
        if np.random.rand() < 0.1:
            deadline = np.random.normal(loc=deadline_offset - 10, scale=3)
        else:
            deadline = np.random.normal(loc=deadline_offset + 10, scale=3)
        return current_time + int(round(deadline))
    if deadline_generation_method == "none":
        return 9999999
    return current_time + int(deadline_offset)


def save_schedule(schedule: NDArray, file_name: str) -> None:
    np.savetxt("data/schedules/" + file_name + ".txt", schedule, fmt="%d")


def save_plots(inbound_frequency: NDArray, outbound_frequency: NDArray, file_name: str) -> None:
    num_skus, total_timesteps = inbound_frequency.shape
    t = np.arange(total_timesteps)
    for sku_idx in range(num_skus):
        plt.figure(figsize=(10, 5))
        plt.plot(t, inbound_frequency[sku_idx], label="Inbound Frequency")
        plt.plot(t, outbound_frequency[sku_idx], label="Outbound Frequency")
        plt.title(f"SKU {sku_idx} Frequency Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Frequency")
        plt.ylim(-1, 2)
        plt.legend()
        plt.grid()
        plt.savefig(f"data/schedules/{file_name}_sku_{sku_idx}_frequency.png")
        plt.close()


def build_functions(args_json: str, total_timesteps: int) -> Tuple[NDArray, NDArray]:
    inbound_frequency = []
    outbound_frequency = []
    sku_functions = json.loads(args_json)
    for sku_func in sku_functions:
        inbound_func = getattr(functions, sku_func["inbound_func"])
        inbound_args = sku_func["inbound_args"]
        outbound_func = getattr(functions, sku_func["outbound_func"])
        outbound_args = sku_func["outbound_args"]

        inbound_frequency.append(inbound_func(*inbound_args, total_timesteps=total_timesteps))
        outbound_frequency.append(outbound_func(*outbound_args, total_timesteps=total_timesteps))

    return np.array(inbound_frequency), np.array(outbound_frequency)


def _load_aisle_locations(map_file: str) -> List[Tuple[int, int]]:
    map_path = "./data/maps/" + map_file
    with open(map_path, "r") as f:
        f.readline()
        f.readline()
        aisle_locations = []
        for i, line in enumerate(f):
            for j, character in enumerate(line):
                if character == "e":
                    aisle_locations.append((i, j))
    return aisle_locations


def replay_inventory(
    time: int,
    initial_counts: Dict[int, int],
    schedule_rows: List[List[int]],
) -> Dict[int, int]:
    """Replay warehouse SKU counts at ``time`` (outbound at release, inbound at deadline)."""
    counts = {sku_id: initial_counts[sku_id] for sku_id in initial_counts}
    for release, deadline, sku_id, task_type in schedule_rows:
        sku_id = int(sku_id)
        if task_type == TASK_TYPE_OUTBOUND and release <= time:
            counts[sku_id] -= 1
        elif task_type == TASK_TYPE_INBOUND and deadline <= time:
            counts[sku_id] += 1
    return counts


def try_generate_task(
    task_type: int,
    release_time: int,
    sku_ids: List[int],
    tasking_weights: Dict[int, float],
    initial_counts: Dict[int, int],
    schedule_rows: List[List[int]],
    num_endpoints: int,
    deadline_generation_method: str,
    deadline_offset: float,
) -> Tuple[bool, int, int]:
    """Return (success, sku_id, deadline). Outbound only uses SKUs with count >= 1."""
    if task_type == TASK_TYPE_OUTBOUND:
        counts = replay_inventory(release_time, initial_counts, schedule_rows)
        eligible = [sku_id for sku_id in sku_ids if counts[sku_id] >= 1]
        if not eligible:
            return False, -1, -1

        eligible_weights = np.array([tasking_weights[sku_id] for sku_id in eligible], dtype=float)
        eligible_weights /= eligible_weights.sum()
        sku_id = int(np.random.choice(eligible, p=eligible_weights))
        deadline = get_deadline(release_time, deadline_generation_method, deadline_offset)
        return True, sku_id, deadline

    counts_at_release = replay_inventory(release_time, initial_counts, schedule_rows)
    if sum(counts_at_release.values()) >= num_endpoints:
        return False, -1, -1

    total_weight = sum(tasking_weights[sku_id] for sku_id in sku_ids)
    weights = [tasking_weights[sku_id] / total_weight for sku_id in sku_ids]
    sku_id = int(np.random.choice(sku_ids, p=weights))
    deadline = get_deadline(release_time, deadline_generation_method, deadline_offset)

    trial_schedule = schedule_rows + [[release_time, deadline, sku_id, TASK_TYPE_INBOUND]]
    counts_at_deadline = replay_inventory(deadline, initial_counts, trial_schedule)
    if sum(counts_at_deadline.values()) > num_endpoints:
        return False, sku_id, deadline
    if any(count < 0 for count in counts_at_deadline.values()):
        return False, sku_id, deadline

    return True, sku_id, deadline


def build_schedules(
    initial_sku_counts: Dict[int, int],
    num_endpoints: int,
    tasking_weights: Dict[int, float],
    initial_inventory_percentage: float,
    tasks_per_timestep: int,
    task_release_frequency: float,
    total_timesteps: int,
    deadline_generation_method: str,
    deadline_offset: float,
) -> NDArray:
    """Generate a feasible schedule using feedback_control inventory logic."""
    sku_ids = list(tasking_weights.keys())
    initial_counts = {sku_id: initial_sku_counts[sku_id] for sku_id in sku_ids}

    k = 1.0 if initial_inventory_percentage <= 50.0 else 0.05
    p_min = 0.15
    p_max = 0.85

    print(f"Number of SKUs: {len(sku_ids)}")
    schedule_rows: List[List[int]] = []

    for t in range(total_timesteps):
        if t % task_release_frequency == 0:
            counts = replay_inventory(t, initial_counts, schedule_rows)
            total_count = sum(counts.values())
            current_inventory = (total_count / num_endpoints) * 100 if num_endpoints > 0 else 0.0
            p_in = np.clip(0.5 + k * (initial_inventory_percentage - current_inventory), p_min, p_max)
            p_out = 1 - p_in
            tasks_to_generate = np.random.choice(
                [TASK_TYPE_OUTBOUND, TASK_TYPE_INBOUND],
                size=tasks_per_timestep,
                p=[p_out, p_in],
            )

            for task_type in tasks_to_generate:
                attempts = 0
                while attempts <= 50:
                    success, sku_id, deadline = try_generate_task(
                        int(task_type),
                        t,
                        sku_ids,
                        tasking_weights,
                        initial_counts,
                        schedule_rows,
                        num_endpoints,
                        deadline_generation_method,
                        deadline_offset,
                    )
                    if success:
                        schedule_rows.append([t, deadline, sku_id, int(task_type)])
                        break
                    attempts += 1

        if t % 100 == 0:
            print(f"Built schedule up to time {t}/{total_timesteps}")

    return np.asarray(schedule_rows, dtype=int)


def build_init_inventory(
    map_file: str,
    init_inventory_full_percentage: float,
    num_skus: int,
    output_file_name: str,
) -> Tuple[Dict[int, int], int, Dict[int, float]]:
    """Initialize warehouse inventory using appearance-weighted placement (mirrors GT Inventory)."""
    aisle_locations = _load_aisle_locations(map_file)
    num_endpoints = len(aisle_locations)
    available_locations = aisle_locations.copy()

    appearance_weights = {sku_id: np.random.uniform(0.1, 1.0) for sku_id in range(num_skus)}
    tasking_weights = {sku_id: np.random.uniform(0.1, 1.0) for sku_id in range(num_skus)}

    total_appearance_weight = sum(appearance_weights.values())
    appearance_probs = [
        appearance_weights[sku_id] / total_appearance_weight for sku_id in range(num_skus)
    ]

    num_init_inventory = int(np.ceil(num_endpoints * init_inventory_full_percentage))
    print(f"Number of SKUs: {num_skus}")
    init_inventory = []
    for _ in range(num_init_inventory):
        if not available_locations:
            break
        loc_idx = np.random.choice(len(available_locations))
        location = available_locations.pop(loc_idx)
        sku_id = int(np.random.choice(num_skus, p=appearance_probs))
        init_inventory.append((sku_id, location[0], location[1]))

    sku_counts = {sku_id: 0 for sku_id in range(num_skus)}
    for sku_id, _, _ in init_inventory:
        sku_counts[sku_id] += 1

    print("Initial Inventory SKU Counts:")
    for sku_id, count in sku_counts.items():
        print(f"SKU {sku_id}: {count}")

    os.makedirs("data/initial_inventories", exist_ok=True)
    np.savetxt(
        "data/initial_inventories/" + output_file_name + "_init_inventory.txt",
        init_inventory,
        fmt="%d",
    )
    np.savetxt(
        "data/initial_inventories/" + output_file_name + "_init_sku_counts.txt",
        [[sku_id, sku_counts[sku_id]] for sku_id in range(num_skus)],
        fmt="%d",
    )

    return sku_counts, num_endpoints, tasking_weights


def validate_schedule(
    schedule: NDArray,
    initial_sku_counts: Dict[int, int],
    num_endpoints: int,
    total_time: int,
) -> None:
    schedule_rows = schedule.tolist()
    maximum_storage_taken = 0
    minimum_storage_taken = 9999

    for t in range(total_time + 1):
        counts = replay_inventory(t, initial_sku_counts, schedule_rows)
        if any(count < 0 for count in counts.values()):
            raise ValueError(f"SKU count became negative at time {t}: {counts}")
        total = sum(counts.values())
        if total > num_endpoints:
            raise ValueError(
                f"Total SKU count {total} exceeded warehouse capacity {num_endpoints} at time {t}"
            )
        maximum_storage_taken = max(maximum_storage_taken, total)
        minimum_storage_taken = min(minimum_storage_taken, total)

    print(f"Maximum storage used during schedule: {maximum_storage_taken}/{num_endpoints}")
    print(f"Minimum storage used during schedule: {minimum_storage_taken}/{num_endpoints}")
    print("Schedule successfully validated.")


def main(
    output_file_name: str,
    map_file_name: str,
    init_inventory_full_percentage: float,
    total_time: int,
    tasks_per_timestep: int,
    task_release_frequency: float,
    num_skus: int,
    args_json: str,
    deadline_generation_method: str,
    deadline_offset: float,
) -> None:
    print("Building Initial Inventory...")
    initial_sku_counts, num_endpoints, tasking_weights = build_init_inventory(
        map_file_name, init_inventory_full_percentage, num_skus, output_file_name
    )
    initial_inventory_percentage = init_inventory_full_percentage * 100.0
    print("Building Schedule (feedback_control)...")
    schedule = build_schedules(
        initial_sku_counts,
        num_endpoints,
        tasking_weights,
        initial_inventory_percentage,
        tasks_per_timestep,
        task_release_frequency,
        total_time,
        deadline_generation_method,
        deadline_offset,
    )

    print(f"Total Tasks: {schedule.shape[0]}")

    for sku_idx in range(num_skus):
        num_inbound_tasks_sku = np.sum(
            (schedule[:, SCHEDULE_SKU_ID] == sku_idx) & (schedule[:, SCHEDULE_TASK_TYPE] == TASK_TYPE_INBOUND)
        )
        num_outbound_tasks_sku = np.sum(
            (schedule[:, SCHEDULE_SKU_ID] == sku_idx) & (schedule[:, SCHEDULE_TASK_TYPE] == TASK_TYPE_OUTBOUND)
        )
        print(f"SKU {sku_idx}: Inbound Tasks: {num_inbound_tasks_sku}, Outbound Tasks: {num_outbound_tasks_sku}")

    validate_schedule(schedule, initial_sku_counts, num_endpoints, total_time)

    num_inbound_tasks = np.sum(schedule[:, SCHEDULE_TASK_TYPE] == TASK_TYPE_INBOUND)
    num_outbound_tasks = np.sum(schedule[:, SCHEDULE_TASK_TYPE] == TASK_TYPE_OUTBOUND)
    print(f"Total Inbound Tasks: {num_inbound_tasks}")
    print(f"Total Outbound Tasks: {num_outbound_tasks}")

    sku_functions = json.loads(args_json)
    if sku_functions:
        inbound_frequency, outbound_frequency = build_functions(args_json, total_timesteps=total_time)
        save_plots(inbound_frequency, outbound_frequency, output_file_name)

    print(schedule)

    os.makedirs("data/schedules", exist_ok=True)
    save_schedule(schedule, output_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct schedule with specified parameters")

    parser.add_argument("--output_file_name", type=str)
    parser.add_argument("--map_file_name", type=str)
    parser.add_argument("--init_inventory_full_percentage", type=float, default=0.5)
    parser.add_argument("--total_time", type=int, default=3600)
    parser.add_argument("--tasks_per_timestep", type=int, default=4)
    parser.add_argument("--task_release_frequency", type=float, default=1.0)
    parser.add_argument("--num_skus", type=int, default=30)
    parser.add_argument("--args_json", type=str)
    parser.add_argument(
        "--deadline-generation-method",
        type=str,
        default="constant",
        choices=["constant", "normal", "bimodal", "none"],
    )
    parser.add_argument(
        "--deadline-offset",
        type=float,
        default=5,
        help="Time between release and deadline (same units as simulation timesteps; 5 = 5 minutes)",
    )
    args = parser.parse_args()

    main(
        output_file_name=args.output_file_name,
        map_file_name=args.map_file_name,
        init_inventory_full_percentage=args.init_inventory_full_percentage,
        total_time=args.total_time,
        tasks_per_timestep=args.tasks_per_timestep,
        task_release_frequency=args.task_release_frequency,
        num_skus=args.num_skus,
        args_json=args.args_json,
        deadline_generation_method=args.deadline_generation_method,
        deadline_offset=args.deadline_offset,
    )
