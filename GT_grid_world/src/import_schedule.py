import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .analysis.statistics import Stats
from .case_request_generator import TASK_TYPE_INBOUND, TASK_TYPE_OUTBOUND, get_deadline
from .graph import Graph

# queue_generation writes 0-indexed SKU ids; GT warehouse SKUs are 1-indexed.
QUEUE_SKU_ID_OFFSET = 1

QUEUE_SKU_ID = 0
QUEUE_TASK_TYPE = 1
QUEUE_DEADLINE = 2


def import_queue(file_path: str) -> NDArray:
    """Load a precomputed task queue (sku_id, task_type[, deadline])."""
    return np.atleast_2d(np.loadtxt(file_path, dtype=int))


def queue_row_deadline(
    task: NDArray,
    current_time: int,
    deadline_generation_method: str,
    deadline_offset: float,
) -> int:
    """Use precomputed queue deadline when present, else generate at release."""
    if task.shape[0] > QUEUE_DEADLINE:
        return int(task[QUEUE_DEADLINE])
    return get_deadline(current_time, deadline_generation_method, deadline_offset)


def load_initial_inventory(
    file_path: str,
    G: Graph,
    sku_id_offset: int = QUEUE_SKU_ID_OFFSET,
) -> None:
    """Load warehouse inventory from queue_generation output (sku_id, row, col)."""
    inventory = np.atleast_2d(np.loadtxt(file_path, dtype=int))
    for sku_id, row, col in inventory:
        G.warehouse.add_sku_instance(int(sku_id) + sku_id_offset, (int(row), int(col)))


def tasks_to_generate_count(frequency: float) -> int:
    """Mirror ``GT_grid_world.execute`` / CRG release batch sizing."""
    if frequency >= 1.0:
        return 1
    if frequency < 1.0:
        return int(frequency**-1)
    return 0


def _add_inbound_task(
    current_time: int,
    sku_id: int,
    J: dict,
    S: Stats,
    G: Graph,
    last_task_id: int,
    deadline_generation_method: str,
    deadline_offset: float,
    deadline: int = None,
) -> Tuple[bool, int]:
    available_start_locations = set(G.driveway.get_empty_locations())
    available_goal_locations = set(G.warehouse.get_empty_locations())

    if not available_start_locations or not available_goal_locations:
        return False, last_task_id

    chosen_start_location = random.choice(list(available_start_locations))
    G.driveway.add_sku_instance(sku_id, chosen_start_location)

    if deadline is None:
        deadline = get_deadline(current_time, deadline_generation_method, deadline_offset)
    last_task_id += 1
    J[last_task_id] = (
        frozenset([chosen_start_location]),
        frozenset(available_goal_locations),
        deadline,
        sku_id,
        TASK_TYPE_INBOUND,
    )
    S.add_task_release(last_task_id, current_time)
    S.add_task_deadline(last_task_id, deadline)
    return True, last_task_id


def _add_outbound_task(
    current_time: int,
    sku_id: int,
    J: dict,
    S: Stats,
    G: Graph,
    last_task_id: int,
    deadline_generation_method: str,
    deadline_offset: float,
    deadline: int = None,
) -> Tuple[bool, int]:
    available_start_locations = set(G.warehouse.get_sku_instances(sku_id))
    available_goal_locations = set(G.driveway.get_empty_locations())

    if not available_start_locations or not available_goal_locations:
        return False, last_task_id

    if deadline is None:
        deadline = get_deadline(current_time, deadline_generation_method, deadline_offset)
    last_task_id += 1
    J[last_task_id] = (
        frozenset(available_start_locations),
        frozenset(available_goal_locations),
        deadline,
        sku_id,
        TASK_TYPE_OUTBOUND,
    )
    S.add_task_release(last_task_id, current_time)
    S.add_task_deadline(last_task_id, deadline)
    return True, last_task_id


def _add_task_from_queue_row(
    current_time: int,
    task: NDArray,
    J: dict,
    S: Stats,
    G: Graph,
    last_task_id: int,
    deadline_generation_method: str,
    deadline_offset: float,
    sku_id_offset: int,
) -> Tuple[bool, int]:
    sku_id = int(task[QUEUE_SKU_ID]) + sku_id_offset
    inbound_outbound = int(task[QUEUE_TASK_TYPE])
    deadline = queue_row_deadline(
        task, current_time, deadline_generation_method, deadline_offset
    )

    if inbound_outbound == TASK_TYPE_INBOUND:
        return _add_inbound_task(
            current_time,
            sku_id,
            J,
            S,
            G,
            last_task_id,
            deadline_generation_method,
            deadline_offset,
            deadline=deadline,
        )

    return _add_outbound_task(
        current_time,
        sku_id,
        J,
        S,
        G,
        last_task_id,
        deadline_generation_method,
        deadline_offset,
        deadline=deadline,
    )


def add_tasks_from_queue(
    current_time: int,
    queue: NDArray,
    deferred_queue: List[List[int]],
    J: dict,
    S: Stats,
    G: Graph,
    last_task_id: int,
    max_task_number: Optional[int],
    frequency: float,
    deadline_generation_method: str,
    deadline_offset: float,
    sku_id_offset: int = QUEUE_SKU_ID_OFFSET,
) -> Tuple[Dict[int, Tuple], List[int], List[int], NDArray, List[List[int]], int]:
    """Release tasks from a precomputed queue into ``J``.

    When ``max_task_number`` is set, pop tasks from the front of the queue
    (and retry deferred tasks first) until ``len(J) >= max_task_number`` or
    no queue rows remain. Deadlines on queue rows are still attached to
    released tasks but do not gate release timing.

    When ``max_task_number`` is ``None`` (legacy cadence), pop up to
    ``tasks_to_generate_count(frequency)`` tasks per tick.

    Tasks that cannot be added are moved to ``deferred_queue``. Each main-queue
    slot consumed always advances the queue, whether or not the task was added.
    """
    outbound_tasks: List[int] = []
    inbound_tasks: List[int] = []

    def at_j_cap() -> bool:
        return max_task_number is not None and len(J) >= max_task_number

    if at_j_cap():
        return J, outbound_tasks, inbound_tasks, queue, deferred_queue, last_task_id

    if queue.size == 0 and not deferred_queue:
        return J, outbound_tasks, inbound_tasks, queue, deferred_queue, last_task_id

    tasks_budget: Optional[int] = (
        None if max_task_number is not None else tasks_to_generate_count(frequency)
    )
    remaining_deferred: List[List[int]] = []

    for i, task_row in enumerate(deferred_queue):
        if at_j_cap():
            remaining_deferred.extend(deferred_queue[i:])
            deferred_queue = remaining_deferred
            return J, outbound_tasks, inbound_tasks, queue, deferred_queue, last_task_id

        success, last_task_id = _add_task_from_queue_row(
            current_time,
            np.asarray(task_row, dtype=int),
            J,
            S,
            G,
            last_task_id,
            deadline_generation_method,
            deadline_offset,
            sku_id_offset,
        )
        if success:
            task_type = int(task_row[QUEUE_TASK_TYPE])
            if task_type == TASK_TYPE_INBOUND:
                inbound_tasks.append(last_task_id)
            else:
                outbound_tasks.append(last_task_id)
            if tasks_budget is not None:
                tasks_budget -= 1
                if tasks_budget <= 0:
                    remaining_deferred.extend(deferred_queue[i + 1 :])
                    deferred_queue = remaining_deferred
                    return J, outbound_tasks, inbound_tasks, queue, deferred_queue, last_task_id
        else:
            remaining_deferred.append(task_row)

    deferred_queue = remaining_deferred

    num_popped = 0
    while num_popped < queue.shape[0]:
        if at_j_cap():
            break
        if tasks_budget is not None and num_popped >= tasks_budget:
            break

        task = queue[num_popped]
        success, last_task_id = _add_task_from_queue_row(
            current_time,
            task,
            J,
            S,
            G,
            last_task_id,
            deadline_generation_method,
            deadline_offset,
            sku_id_offset,
        )

        if success:
            inbound_outbound = int(task[QUEUE_TASK_TYPE])
            if inbound_outbound == TASK_TYPE_INBOUND:
                inbound_tasks.append(last_task_id)
            else:
                outbound_tasks.append(last_task_id)
        else:
            deferred_queue.append(task.tolist())

        num_popped += 1

    if num_popped:
        queue = queue[num_popped:]
        if queue.size == 0:
            cols = queue.shape[1] if queue.ndim > 1 else 2
            queue = np.empty((0, cols), dtype=int)
        else:
            queue = np.atleast_2d(queue)

    return J, outbound_tasks, inbound_tasks, queue, deferred_queue, last_task_id


def queue_tasks_finished(
    queue: NDArray,
    deferred_queue: List[List[int]],
    J: dict,
    S: Stats,
    total_queue_tasks: int,
) -> bool:
    """True when every queue task has been released, completed, and ``J`` is empty."""
    if queue.size > 0 or deferred_queue or len(J) > 0:
        return False
    completed_ids = set(S.get_completed_task_ids())
    return len(completed_ids) >= total_queue_tasks
