import numpy as np
import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

from numpy.typing import NDArray

import functions

TASK_TYPE_OUTBOUND = 0
TASK_TYPE_INBOUND = 1

QUEUE_SKU_ID = 0
QUEUE_TASK_TYPE = 1
QUEUE_DEADLINE = 2


def save_queue(queue: NDArray, file_name: str) -> None:
    np.savetxt("data/queues/" + file_name + ".txt", queue, fmt="%d")


def tasks_per_second(deadline_rate: float) -> float:
    """Convert tasks/min release rate to tasks per simulated second."""
    if deadline_rate <= 0:
        raise ValueError(f"deadline_rate must be positive, got {deadline_rate}")
    return deadline_rate / 60.0


def deadline_second_for_task_index(task_index: int, deadline_rate: float) -> int:
    """Return the 1-based deadline second for a zero-based queue task index.

    At ``deadline_rate=60``, tasks release one per second (00:01, 00:02, …).
    At ``deadline_rate=120``, two tasks per second on average (00:01, 00:01, …).
    At ``deadline_rate=90``, 1.5 tasks per second on average (some seconds
    carry one task, others two).
    """
    return int(np.ceil((task_index + 1) * 60.0 / deadline_rate))


def assign_queue_deadlines(number_of_tasks: int, deadline_rate: float) -> NDArray:
    """Return 1-based deadline timesteps (seconds) for each queue index."""
    if number_of_tasks <= 0:
        return np.array([], dtype=int)
    indices = np.arange(number_of_tasks, dtype=int)
    return np.ceil((indices + 1) * 60.0 / deadline_rate).astype(int)


def expected_queue_duration_seconds(number_of_tasks: int, deadline_rate: float) -> int:
    """Wall-clock seconds spanned by ``number_of_tasks`` at ``deadline_rate`` tasks/min."""
    if number_of_tasks <= 0:
        return 0
    return deadline_second_for_task_index(number_of_tasks - 1, deadline_rate)


def format_deadline_seconds(seconds: int) -> str:
    """Format a 1-based deadline second as mm:ss (e.g. 1 -> 00:01)."""
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def save_weight_plots(weight_profiles: NDArray, file_name: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Skipping weight plots: matplotlib is not installed")
        return

    num_skus, total_tasks = weight_profiles.shape
    t = np.arange(total_tasks)
    for sku_idx in range(num_skus):
        plt.figure(figsize=(10, 5))
        plt.plot(t, weight_profiles[sku_idx], label="Tasking Weight")
        plt.title(f"SKU {sku_idx} Tasking Weight Over Queue Generation")
        plt.xlabel("Task Index")
        plt.ylabel("Weight")
        plt.legend()
        plt.grid()
        plt.savefig(f"data/queues/{file_name}_sku_{sku_idx}_weight.png")
        plt.close()


def _resolve_weight_function(weight_func_name: str):
    aliases = {
        "constant": functions.constant_weight,
        "sinusoid": functions.sinusoid_weight,
        "sinusoidal": functions.sinusoid_weight,
    }
    if weight_func_name not in aliases:
        raise ValueError(
            f"Unknown weight_func {weight_func_name!r}; expected one of {sorted(aliases)}"
        )
    return aliases[weight_func_name]


def build_sku_weight_profiles(
    num_skus: int,
    number_of_tasks: int,
    weight_mode: str,
    sku_weights_json: str,
) -> NDArray:
    """Build per-SKU tasking weights over queue generation indices."""
    profiles = np.ones((num_skus, number_of_tasks), dtype=float)

    if weight_mode == "uniform":
        return profiles
    if weight_mode != "custom":
        raise ValueError(f"Unknown weight_mode {weight_mode!r}; expected 'uniform' or 'custom'")

    sku_configs = json.loads(sku_weights_json)
    if isinstance(sku_configs, dict):
        sku_configs = sku_configs.get("profiles", [])
    if not sku_configs:
        return profiles

    assigned_skus = set()
    for entry in sku_configs:
        sku_ids = entry["sku_ids"]
        weight_func_name = entry["weight_func"]
        weight_func = _resolve_weight_function(weight_func_name)
        weight_args = entry["weight_args"]

        if weight_func_name in ("sinusoid", "sinusoidal") and len(weight_args) != 4:
            raise ValueError(
                "sinusoid weight_args must be [min_weight, max_weight, period_tasks, phase]"
            )

        profile = weight_func(*weight_args, total_timesteps=number_of_tasks)

        for sku_id in sku_ids:
            sku_id = int(sku_id)
            if sku_id < 0 or sku_id >= num_skus:
                raise ValueError(f"sku_id {sku_id} out of range for num_skus={num_skus}")
            if sku_id in assigned_skus:
                raise ValueError(f"sku_id {sku_id} appears in more than one weight config entry")
            profiles[sku_id, :] = profile
            assigned_skus.add(sku_id)

    return profiles


def _tasking_weights_at_index(
    task_idx: int,
    sku_ids: List[int],
    tasking_weight_profiles: NDArray,
) -> Dict[int, float]:
    return {sku_id: float(tasking_weight_profiles[sku_id, task_idx]) for sku_id in sku_ids}


def save_plots(inbound_frequency: NDArray, outbound_frequency: NDArray, file_name: str) -> None:
    import matplotlib.pyplot as plt

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
        plt.savefig(f"data/queues/{file_name}_sku_{sku_idx}_frequency.png")
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


def _location_depth(
    location: Tuple[int, int],
    aisle_locations: List[Tuple[int, int]],
) -> float:
    """Return depth in [0, 1], where 0 is the back and 1 is the front (near driveway)."""
    rows = [row for row, _ in aisle_locations]
    min_row, max_row = min(rows), max(rows)
    if max_row == min_row:
        return 0.5
    row, _ = location
    return (row - min_row) / (max_row - min_row)


def _normalize_sku_weights(sku_tasking_weights: Dict[int, float]) -> Dict[int, float]:
    values = list(sku_tasking_weights.values())
    lo, hi = min(values), max(values)
    if hi <= lo:
        return {sku_id: 0.5 for sku_id in sku_tasking_weights}
    return {sku_id: (weight - lo) / (hi - lo) for sku_id, weight in sku_tasking_weights.items()}


def _mean_sku_tasking_weights(tasking_weight_profiles: NDArray) -> Dict[int, float]:
    return {
        sku_id: float(tasking_weight_profiles[sku_id].mean())
        for sku_id in range(tasking_weight_profiles.shape[0])
    }


def _sample_uniform_inventory(
    num_init_inventory: int,
    num_skus: int,
    available_locations: List[Tuple[int, int]],
    appearance_probs: List[float],
) -> List[Tuple[int, int, int]]:
    init_inventory: List[Tuple[int, int, int]] = []
    for _ in range(num_init_inventory):
        if not available_locations:
            break
        loc_idx = np.random.choice(len(available_locations))
        location = available_locations.pop(loc_idx)
        sku_id = int(np.random.choice(num_skus, p=appearance_probs))
        init_inventory.append((sku_id, location[0], location[1]))
    return init_inventory


def _sample_adversarial_inventory(
    num_init_inventory: int,
    num_skus: int,
    aisle_locations: List[Tuple[int, int]],
    available_locations: List[Tuple[int, int]],
    appearance_probs: List[float],
    sku_tasking_weights: Dict[int, float],
) -> List[Tuple[int, int, int]]:
    """Place high-weight SKUs toward the back and low-weight SKUs toward the front."""
    normalized_weights = _normalize_sku_weights(sku_tasking_weights)
    init_inventory: List[Tuple[int, int, int]] = []

    for _ in range(num_init_inventory):
        if not available_locations:
            break

        sku_id = int(np.random.choice(num_skus, p=appearance_probs))
        sku_weight = normalized_weights[sku_id]

        location_weights = []
        for location in available_locations:
            depth = _location_depth(location, aisle_locations)
            affinity = sku_weight * (1.0 - depth) + (1.0 - sku_weight) * depth
            location_weights.append(max(affinity, 1e-6))

        location_probs = np.array(location_weights, dtype=float)
        location_probs /= location_probs.sum()
        loc_idx = int(np.random.choice(len(available_locations), p=location_probs))
        location = available_locations.pop(loc_idx)
        init_inventory.append((sku_id, location[0], location[1]))

    return init_inventory


def _apply_task_to_tally(tally: Dict[int, int], sku_id: int, task_type: int) -> int:
    """Update projected inventory tally in place. Returns change in total count."""
    sku_id = int(sku_id)
    if task_type == TASK_TYPE_OUTBOUND:
        tally[sku_id] -= 1
        return -1
    tally[sku_id] += 1
    return 1


def try_generate_task(
    task_type: int,
    task_idx: int,
    sku_ids: List[int],
    tasking_weight_profiles: NDArray,
    tally: Dict[int, int],
    total_count: int,
    num_endpoints: int,
) -> Tuple[bool, int]:
    """Return (success, sku_id). Uses running inventory tally (initial + queued tasks)."""
    tasking_weights = _tasking_weights_at_index(task_idx, sku_ids, tasking_weight_profiles)

    if task_type == TASK_TYPE_OUTBOUND:
        eligible = [sku_id for sku_id in sku_ids if tally[sku_id] >= 1]
        if not eligible:
            return False, -1

        eligible_weights = np.array([tasking_weights[sku_id] for sku_id in eligible], dtype=float)
        eligible_weights /= eligible_weights.sum()
        sku_id = int(np.random.choice(eligible, p=eligible_weights))
        return True, sku_id

    if total_count >= num_endpoints:
        return False, -1

    total_weight = sum(tasking_weights[sku_id] for sku_id in sku_ids)
    weights = [tasking_weights[sku_id] / total_weight for sku_id in sku_ids]
    sku_id = int(np.random.choice(sku_ids, p=weights))

    if total_count + 1 > num_endpoints:
        return False, sku_id
    if tally[sku_id] + 1 < 0:
        return False, sku_id

    return True, sku_id


def build_queue(
    initial_sku_counts: Dict[int, int],
    num_endpoints: int,
    tasking_weight_profiles: NDArray,
    initial_inventory_percentage: float,
    number_of_tasks: int,
    deadline_rate: float,
) -> NDArray:
    """Generate a feasible task queue using feedback_control inventory logic."""
    sku_ids = list(range(tasking_weight_profiles.shape[0]))
    initial_counts = {sku_id: initial_sku_counts[sku_id] for sku_id in sku_ids}

    k = 1.0 if initial_inventory_percentage <= 50.0 else 0.05
    p_min = 0.15
    p_max = 0.85

    print(f"Number of SKUs: {len(sku_ids)}")
    rate_per_sec = tasks_per_second(deadline_rate)
    duration_sec = expected_queue_duration_seconds(number_of_tasks, deadline_rate)
    print(
        f"Deadline rate: {deadline_rate} tasks/min "
        f"({rate_per_sec:.4g} tasks/s, ~{duration_sec}s for {number_of_tasks} tasks)"
    )
    queue_rows: List[List[int]] = []
    tally = {sku_id: initial_counts[sku_id] for sku_id in sku_ids}
    total_count = sum(tally.values())

    for task_idx in range(number_of_tasks):
        current_inventory = (total_count / num_endpoints) * 100 if num_endpoints > 0 else 0.0
        p_in = np.clip(0.5 + k * (initial_inventory_percentage - current_inventory), p_min, p_max)
        p_out = 1 - p_in
        task_type = int(np.random.choice([TASK_TYPE_OUTBOUND, TASK_TYPE_INBOUND], p=[p_out, p_in]))

        attempts = 0
        while attempts <= 50:
            success, sku_id = try_generate_task(
                task_type,
                task_idx,
                sku_ids,
                tasking_weight_profiles,
                tally,
                total_count,
                num_endpoints,
            )
            if success:
                deadline = deadline_second_for_task_index(len(queue_rows), deadline_rate)
                queue_rows.append([sku_id, task_type, deadline])
                total_count += _apply_task_to_tally(tally, sku_id, task_type)
                break
            attempts += 1

        if task_idx % 1000 == 0:
            print(f"Built queue: {len(queue_rows)}/{number_of_tasks} tasks generated")

    return np.asarray(queue_rows, dtype=int)


def build_init_inventory(
    map_file: str,
    init_inventory_full_percentage: float,
    num_skus: int,
    output_file_name: str,
    inventory_layout_mode: str = "uniform",
    sku_tasking_weights: Optional[Dict[int, float]] = None,
) -> Tuple[Dict[int, int], int]:
    """Initialize warehouse inventory placement."""
    if inventory_layout_mode not in ("uniform", "adversarial"):
        raise ValueError(
            f"Unknown inventory_layout_mode {inventory_layout_mode!r}; "
            "expected 'uniform' or 'adversarial'"
        )
    if inventory_layout_mode == "adversarial" and sku_tasking_weights is None:
        raise ValueError("adversarial inventory layout requires sku_tasking_weights")

    aisle_locations = _load_aisle_locations(map_file)
    num_endpoints = len(aisle_locations)
    available_locations = aisle_locations.copy()

    appearance_weights = {sku_id: np.random.uniform(0.1, 1.0) for sku_id in range(num_skus)}
    total_appearance_weight = sum(appearance_weights.values())
    appearance_probs = [
        appearance_weights[sku_id] / total_appearance_weight for sku_id in range(num_skus)
    ]

    num_init_inventory = int(np.ceil(num_endpoints * init_inventory_full_percentage))
    print(f"Number of SKUs: {num_skus}")
    print(f"Inventory layout mode: {inventory_layout_mode}")

    if inventory_layout_mode == "uniform":
        init_inventory = _sample_uniform_inventory(
            num_init_inventory, num_skus, available_locations, appearance_probs
        )
    else:
        init_inventory = _sample_adversarial_inventory(
            num_init_inventory,
            num_skus,
            aisle_locations,
            available_locations,
            appearance_probs,
            sku_tasking_weights,
        )

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

    return sku_counts, num_endpoints


def validate_queue(
    queue: NDArray,
    initial_sku_counts: Dict[int, int],
    num_endpoints: int,
) -> None:
    """Check inventory stays feasible after each prefix of the queue."""
    tally = {sku_id: initial_sku_counts[sku_id] for sku_id in initial_sku_counts}
    total = sum(tally.values())
    maximum_storage_taken = total
    minimum_storage_taken = total

    for row in queue.tolist():
        sku_id, task_type = int(row[QUEUE_SKU_ID]), int(row[QUEUE_TASK_TYPE])
        total += _apply_task_to_tally(tally, sku_id, task_type)
        if any(count < 0 for count in tally.values()):
            raise ValueError(f"SKU count became negative after queue prefix: {tally}")
        if total > num_endpoints:
            raise ValueError(
                f"Total SKU count {total} exceeded warehouse capacity {num_endpoints} "
                f"after queue prefix"
            )
        maximum_storage_taken = max(maximum_storage_taken, total)
        minimum_storage_taken = min(minimum_storage_taken, total)

    print(f"Maximum projected storage after queue prefixes: {maximum_storage_taken}/{num_endpoints}")
    print(f"Minimum projected storage after queue prefixes: {minimum_storage_taken}/{num_endpoints}")
    print("Queue successfully validated.")


def main(
    seed: int,
    output_file_name: str,
    map_file_name: str,
    init_inventory_full_percentage: float,
    number_of_tasks: int,
    num_skus: int,
    weight_mode: str,
    sku_weights_json: str,
    inventory_layout_mode: str,
    save_plots: bool,
    args_json: str,
    deadline_rate: float,
) -> None:
    np.random.seed(seed)
    print(f"Random seed: {seed}")

    tasking_weight_profiles = build_sku_weight_profiles(
        num_skus, number_of_tasks, weight_mode, sku_weights_json
    )
    sku_tasking_weights = _mean_sku_tasking_weights(tasking_weight_profiles)

    print("Building Initial Inventory...")
    initial_sku_counts, num_endpoints = build_init_inventory(
        map_file_name,
        init_inventory_full_percentage,
        num_skus,
        output_file_name,
        inventory_layout_mode=inventory_layout_mode,
        sku_tasking_weights=sku_tasking_weights,
    )
    initial_inventory_percentage = init_inventory_full_percentage * 100.0
    print(f"Building Queue (feedback_control, weight_mode={weight_mode})...")
    queue = build_queue(
        initial_sku_counts,
        num_endpoints,
        tasking_weight_profiles,
        initial_inventory_percentage,
        number_of_tasks,
        deadline_rate,
    )

    print(f"Total Tasks: {queue.shape[0]} (target: {number_of_tasks})")
    if queue.shape[0] > 0:
        last_deadline = int(queue[-1, QUEUE_DEADLINE])
        print(
            f"Queue spans {last_deadline}s "
            f"({format_deadline_seconds(last_deadline)}) at {deadline_rate} tasks/min"
        )
        sample = min(6, queue.shape[0])
        print("First queue deadlines (mm:ss):")
        for row in queue[:sample]:
            print(f"  task -> {format_deadline_seconds(int(row[QUEUE_DEADLINE]))}")
        if queue.shape[0] >= 60:
            last_in_minute = queue[59, QUEUE_DEADLINE]
            print(f"Deadline at queue index 59: {format_deadline_seconds(int(last_in_minute))}")

    for sku_idx in range(num_skus):
        num_inbound_tasks_sku = np.sum(
            (queue[:, QUEUE_SKU_ID] == sku_idx) & (queue[:, QUEUE_TASK_TYPE] == TASK_TYPE_INBOUND)
        )
        num_outbound_tasks_sku = np.sum(
            (queue[:, QUEUE_SKU_ID] == sku_idx) & (queue[:, QUEUE_TASK_TYPE] == TASK_TYPE_OUTBOUND)
        )
        print(f"SKU {sku_idx}: Inbound Tasks: {num_inbound_tasks_sku}, Outbound Tasks: {num_outbound_tasks_sku}")

    validate_queue(queue, initial_sku_counts, num_endpoints)

    num_inbound_tasks = np.sum(queue[:, QUEUE_TASK_TYPE] == TASK_TYPE_INBOUND)
    num_outbound_tasks = np.sum(queue[:, QUEUE_TASK_TYPE] == TASK_TYPE_OUTBOUND)
    print(f"Total Inbound Tasks: {num_inbound_tasks}")
    print(f"Total Outbound Tasks: {num_outbound_tasks}")

    if save_plots and weight_mode == "custom":
        save_weight_plots(tasking_weight_profiles, output_file_name)

    sku_functions = json.loads(args_json)
    if save_plots and sku_functions:
        inbound_frequency, outbound_frequency = build_functions(args_json, total_timesteps=number_of_tasks)
        save_plots(inbound_frequency, outbound_frequency, output_file_name)

    print(queue)

    os.makedirs("data/queues", exist_ok=True)
    save_queue(queue, output_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct task queue with specified parameters")

    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--output_file_name", type=str)
    parser.add_argument("--map_file_name", type=str)
    parser.add_argument("--init_inventory_full_percentage", type=float, default=0.5)
    parser.add_argument("--number_of_tasks", type=int, default=7200)
    parser.add_argument("--num_skus", type=int, default=30)
    parser.add_argument(
        "--weight_mode",
        type=str,
        default="uniform",
        choices=["uniform", "custom"],
        help="SKU tasking weight mode: uniform (equal weights) or custom (from JSON config)",
    )
    parser.add_argument(
        "--sku_weights_json",
        type=str,
        default="[]",
        help='JSON list of per-SKU weight configs. constant: [value]. sinusoid: [min_weight, max_weight, period_tasks, phase].',
    )
    parser.add_argument(
        "--inventory_layout_mode",
        type=str,
        default="uniform",
        choices=["uniform", "adversarial"],
        help="Initial inventory placement: uniform (random locations) or adversarial "
             "(high-weight SKUs toward back, low-weight toward front)",
    )
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-SKU weight/frequency plots to data/queues/ (default: true)",
    )
    parser.add_argument(
        "--args_json",
        type=str,
        default="[]",
        help="Optional legacy JSON for inbound/outbound frequency plots",
    )
    parser.add_argument(
        "--deadline-rate",
        type=float,
        default=60.0,
        help="Task release rate in tasks/min. Deadlines are 1-based simulated "
             "seconds computed as ceil((task_index + 1) * 60 / rate). "
             "Examples: 60 -> 5000 tasks over 5000s; 120 -> 2500s; 90 -> ~3334s.",
    )
    args = parser.parse_args()

    main(
        seed=args.seed,
        output_file_name=args.output_file_name,
        map_file_name=args.map_file_name,
        init_inventory_full_percentage=args.init_inventory_full_percentage,
        number_of_tasks=args.number_of_tasks,
        num_skus=args.num_skus,
        weight_mode=args.weight_mode,
        sku_weights_json=args.sku_weights_json,
        inventory_layout_mode=args.inventory_layout_mode,
        save_plots=args.save_plots,
        args_json=args.args_json,
        deadline_rate=args.deadline_rate,
    )
