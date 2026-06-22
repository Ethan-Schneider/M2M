import random
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from .analysis.statistics import Stats
from .graph import Graph

# schedule_generation writes 0-indexed SKU ids; GT warehouse SKUs are 1-indexed.
SCHEDULE_SKU_ID_OFFSET = 1

SCHEDULE_RELEASE_TIME = 0
SCHEDULE_DEADLINE = 1
SCHEDULE_SKU_ID = 2
SCHEDULE_TASK_TYPE = 3


def import_schedule(file_path: str) -> NDArray:
    """Load a precomputed schedule from a text file."""
    return np.atleast_2d(np.loadtxt(file_path, dtype=int))


def load_initial_inventory(file_path: str, G: Graph, sku_id_offset: int = SCHEDULE_SKU_ID_OFFSET) -> None:
    """Load warehouse inventory from schedule_generation output (sku_id, row, col)."""
    inventory = np.atleast_2d(np.loadtxt(file_path, dtype=int))
    for sku_id, row, col in inventory:
        G.warehouse.add_sku_instance(int(sku_id) + sku_id_offset, (int(row), int(col)))


def _remove_tasks_from_schedule(schedule: NDArray, added_rows: List[NDArray]) -> NDArray:
    if schedule.size == 0 or not added_rows:
        return schedule
    mask = np.ones(schedule.shape[0], dtype=bool)
    for row in added_rows:
        matches = np.all(schedule == row, axis=1)
        remaining_matches = np.flatnonzero(matches & mask)
        if remaining_matches.size == 0:
            continue
        mask[remaining_matches[0]] = False
    remaining = schedule[mask]
    if remaining.size == 0:
        return np.empty((0, 4), dtype=int)
    return np.atleast_2d(remaining)


def _due_schedule_tasks(schedule: NDArray, current_time: int) -> NDArray:
    """Return schedule rows whose release time has passed, oldest first."""
    if schedule.size == 0:
        return schedule
    due_mask = schedule[:, SCHEDULE_RELEASE_TIME] <= current_time
    due_tasks = schedule[due_mask]
    if due_tasks.size == 0:
        return due_tasks
    order = np.argsort(due_tasks[:, SCHEDULE_RELEASE_TIME], kind="stable")
    return due_tasks[order]


def add_tasks_from_schedule(
    current_time: int,
    schedule: NDArray,
    J: dict,
    S: Stats,
    G: Graph,
    last_task_id: int,
    sku_id_offset: int = SCHEDULE_SKU_ID_OFFSET,
) -> Tuple[Dict[int, Tuple], List[int], List[int], NDArray, int]:
    """Append due schedule tasks (release_time <= current_time) into ``J``."""
    tasks_to_try = _due_schedule_tasks(schedule, current_time)

    outbound_tasks: List[int] = []
    inbound_tasks: List[int] = []
    added_rows: List[NDArray] = []

    if tasks_to_try.size == 0:
        return J, outbound_tasks, inbound_tasks, schedule, last_task_id

    for task in tasks_to_try:
        release_time = int(task[SCHEDULE_RELEASE_TIME])
        deadline = int(task[SCHEDULE_DEADLINE])
        sku_id = int(task[SCHEDULE_SKU_ID]) + sku_id_offset
        inbound_outbound = int(task[SCHEDULE_TASK_TYPE])

        if inbound_outbound == 1:
            available_start_locations = set(G.driveway.get_empty_locations())
            available_goal_locations = set(G.warehouse.get_empty_locations())

            if not available_start_locations or not available_goal_locations:
                continue

            chosen_start_location = random.choice(list(available_start_locations))
            G.driveway.add_sku_instance(sku_id, chosen_start_location)

            last_task_id += 1
            J[last_task_id] = (
                frozenset([chosen_start_location]),
                frozenset(available_goal_locations),
                deadline,
                sku_id,
                1,
            )
            S.add_task_release(last_task_id, release_time)
            S.add_task_deadline(last_task_id, deadline)
            inbound_tasks.append(last_task_id)
            added_rows.append(task)
        else:
            available_start_locations = set(G.warehouse.get_sku_instances(sku_id))
            available_goal_locations = set(G.driveway.get_empty_locations())

            if not available_start_locations or not available_goal_locations:
                continue

            last_task_id += 1
            J[last_task_id] = (
                frozenset(available_start_locations),
                frozenset(available_goal_locations),
                deadline,
                sku_id,
                0,
            )
            S.add_task_release(last_task_id, release_time)
            S.add_task_deadline(last_task_id, deadline)
            outbound_tasks.append(last_task_id)
            added_rows.append(task)

    schedule = _remove_tasks_from_schedule(schedule, added_rows)

    return J, outbound_tasks, inbound_tasks, schedule, last_task_id


def schedule_tasks_finished(
    schedule: NDArray,
    J: dict,
    S: Stats,
    total_schedule_tasks: int,
) -> bool:
    """True when every schedule task has been released, completed, and ``J`` is empty."""
    if schedule.size > 0 or len(J) > 0:
        return False
    completed_ids = set(S.get_completed_task_ids())
    return len(completed_ids) >= total_schedule_tasks
