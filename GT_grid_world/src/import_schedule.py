import random
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from .analysis.statistics import Stats
from .case_request_generator import TASK_TYPE_INBOUND, TASK_TYPE_OUTBOUND, get_deadline
from .graph import Graph

# schedule_generation writes 0-indexed SKU ids; GT warehouse SKUs are 1-indexed.
SCHEDULE_SKU_ID_OFFSET = 1

SCHEDULE_RELEASE_TIME = 0
SCHEDULE_DEADLINE = 1
SCHEDULE_SKU_ID = 2
SCHEDULE_TASK_TYPE = 3

# queue_generation writes 0-indexed SKU ids; GT warehouse SKUs are 1-indexed.
QUEUE_SKU_ID_OFFSET = 1

# A queue row is just (sku_id, task_type) -- there is no baked-in release time.
# Tasks are pulled on demand (see ``add_tasks_from_queue``), which is what makes
# arrival demand-driven instead of the schedule's time-stamped dumps.
QUEUE_SKU_ID = 0
QUEUE_TASK_TYPE = 1


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
        if not np.any(matches):
            continue
        mask[np.argmax(matches)] = False
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
    deterministic: bool = False,
) -> Tuple[Dict[int, Tuple], List[int], List[int], NDArray, int]:
    """Append due schedule tasks (release_time <= current_time) into ``J``.

    ``deterministic`` makes the inbound driveway-cell selection
    reproducible: the lowest-(row, col) *empty* driveway cell is
    picked instead of ``random.choice``. Under TA-Hybrid the
    driver re-syncs each agent's pre-committed pickup cell to the
    cell actually chosen here once the task lands in ``J``. Default
    ``False`` preserves the legacy random-cell behaviour for every
    other allocator.
    """
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

            if deterministic:
                chosen_start_location = min(available_start_locations)
            else:
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
    S: Stats,
    total_schedule_tasks: int,
) -> bool:
    """True when every schedule task has been released and completed."""
    if schedule.size > 0:
        return False
    completed_ids = set(S.get_completed_task_ids())
    return len(completed_ids) >= total_schedule_tasks


# ---------------------------------------------------------------------------
# Queue-based (demand-driven) task arrival
#
# Unlike a schedule -- whose rows carry release times and are dumped into ``J``
# when ``release_time <= t`` -- a queue is an ordered list of ``(sku_id,
# task_type)`` rows with no timing. Tasks are pulled on demand: at most a
# frequency-sized batch per release tick, and only while ``len(J) <
# max_task_number``. Rows that cannot be placed (no empty start/goal cell right
# now) are parked in a ``deferred_queue`` and retried first on the next tick.
# This produces backpressure -- the system pulls "however many it needs" as
# tasks drain -- instead of releasing a fixed burst every minute.
# ---------------------------------------------------------------------------


def import_queue(file_path: str) -> NDArray:
    """Load a precomputed task queue from a text file (sku_id, task_type)."""
    return np.atleast_2d(np.loadtxt(file_path, dtype=int))


def tasks_to_generate_count(frequency: float) -> int:
    """Per-release-tick task budget, mirroring CRG / schedule release sizing."""
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
) -> Tuple[bool, int]:
    available_start_locations = set(G.driveway.get_empty_locations())
    available_goal_locations = set(G.warehouse.get_empty_locations())

    if not available_start_locations or not available_goal_locations:
        return False, last_task_id

    chosen_start_location = random.choice(list(available_start_locations))
    G.driveway.add_sku_instance(sku_id, chosen_start_location)

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
) -> Tuple[bool, int]:
    available_start_locations = set(G.warehouse.get_sku_instances(sku_id))
    available_goal_locations = set(G.driveway.get_empty_locations())

    if not available_start_locations or not available_goal_locations:
        return False, last_task_id

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

    if inbound_outbound == TASK_TYPE_INBOUND:
        return _add_inbound_task(
            current_time, sku_id, J, S, G, last_task_id,
            deadline_generation_method, deadline_offset,
        )

    return _add_outbound_task(
        current_time, sku_id, J, S, G, last_task_id,
        deadline_generation_method, deadline_offset,
    )


def add_tasks_from_queue(
    current_time: int,
    queue: NDArray,
    deferred_queue: List[List[int]],
    J: dict,
    S: Stats,
    G: Graph,
    last_task_id: int,
    max_task_number: int,
    frequency: float,
    deadline_generation_method: str,
    deadline_offset: float,
    sku_id_offset: int = QUEUE_SKU_ID_OFFSET,
) -> Tuple[Dict[int, Tuple], List[int], List[int], NDArray, List[List[int]], int]:
    """Pop up to a frequency-sized batch of queue tasks into ``J`` on demand.

    Release is gated by ``len(J) < max_task_number`` (backpressure) and the
    per-tick budget from ``tasks_to_generate_count``. Rows that cannot be placed
    right now (no free start/goal cell) are moved to ``deferred_queue``; deferred
    rows are retried first on the next release tick before new rows are popped.
    Successful deferred adds count toward the per-tick budget; failed attempts do
    not. Each main-queue slot consumed always advances the queue, whether or not
    the task was added.
    """
    outbound_tasks: List[int] = []
    inbound_tasks: List[int] = []

    if len(J) >= max_task_number:
        return J, outbound_tasks, inbound_tasks, queue, deferred_queue, last_task_id

    if queue.size == 0 and not deferred_queue:
        return J, outbound_tasks, inbound_tasks, queue, deferred_queue, last_task_id

    tasks_budget = tasks_to_generate_count(frequency)
    remaining_deferred: List[List[int]] = []

    for i, task_row in enumerate(deferred_queue):
        if len(J) >= max_task_number:
            remaining_deferred.extend(deferred_queue[i:])
            deferred_queue = remaining_deferred
            return J, outbound_tasks, inbound_tasks, queue, deferred_queue, last_task_id

        success, last_task_id = _add_task_from_queue_row(
            current_time, np.asarray(task_row, dtype=int), J, S, G, last_task_id,
            deadline_generation_method, deadline_offset, sku_id_offset,
        )
        if success:
            if int(task_row[QUEUE_TASK_TYPE]) == TASK_TYPE_INBOUND:
                inbound_tasks.append(last_task_id)
            else:
                outbound_tasks.append(last_task_id)
            tasks_budget -= 1
            if tasks_budget <= 0:
                remaining_deferred.extend(deferred_queue[i + 1:])
                deferred_queue = remaining_deferred
                return J, outbound_tasks, inbound_tasks, queue, deferred_queue, last_task_id
        else:
            remaining_deferred.append(task_row)

    deferred_queue = remaining_deferred

    main_slots = tasks_budget
    num_popped = 0

    for _ in range(main_slots):
        if queue.size == 0 or len(J) >= max_task_number:
            break

        task = queue[num_popped]
        success, last_task_id = _add_task_from_queue_row(
            current_time, task, J, S, G, last_task_id,
            deadline_generation_method, deadline_offset, sku_id_offset,
        )

        if success:
            if int(task[QUEUE_TASK_TYPE]) == TASK_TYPE_INBOUND:
                inbound_tasks.append(last_task_id)
            else:
                outbound_tasks.append(last_task_id)
        else:
            deferred_queue.append(task.tolist())

        num_popped += 1

    if num_popped:
        queue = queue[num_popped:]
        if queue.size == 0:
            queue = np.empty((0, 2), dtype=int)
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
