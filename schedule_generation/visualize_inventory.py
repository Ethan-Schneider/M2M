import numpy as np
import matplotlib.pyplot as plt

SCHEDULE_RELEASE_TIME = 0
SCHEDULE_DEADLINE = 1
SCHEDULE_SKU_ID = 2
SCHEDULE_TASK_TYPE = 3
TASK_TYPE_OUTBOUND = 0
TASK_TYPE_INBOUND = 1


def _num_skus(initial_inventory: np.ndarray, schedule: np.ndarray) -> int:
    initial_inventory = np.atleast_2d(initial_inventory)
    schedule = np.atleast_2d(schedule)
    max_sku_id = -1
    if initial_inventory.size:
        max_sku_id = max(max_sku_id, int(initial_inventory[:, 0].max()))
    if schedule.size:
        max_sku_id = max(max_sku_id, int(schedule[:, SCHEDULE_SKU_ID].max()))
    return max_sku_id + 1


def replay_inventory(time: int, initial_counts: dict, schedule: np.ndarray) -> dict:
    counts = dict(initial_counts)
    for release, deadline, sku_id, task_type in schedule.astype(int):
        if task_type == TASK_TYPE_OUTBOUND and release <= time:
            counts[sku_id] -= 1
        elif task_type == TASK_TYPE_INBOUND and deadline <= time:
            counts[sku_id] += 1
    return counts


def visualize_inventory(initial_inventory: np.ndarray, schedule: np.ndarray) -> None:
    initial_inventory = np.atleast_2d(np.asarray(initial_inventory))
    schedule = np.atleast_2d(np.asarray(schedule)).astype(int)

    num_skus = _num_skus(initial_inventory, schedule)
    initial_counts = {sku_id: 0 for sku_id in range(num_skus)}
    for sku_id, _, _ in initial_inventory:
        initial_counts[int(sku_id)] += 1

    if schedule.size == 0:
        max_time = 0
    else:
        max_time = int(
            max(
                schedule[:, SCHEDULE_RELEASE_TIME].max(),
                schedule[:, SCHEDULE_DEADLINE].max(),
            )
        )

    times = list(range(max_time + 1))
    total_counts = [sum(replay_inventory(t, initial_counts, schedule).values()) for t in times]

    plt.figure(figsize=(10, 5))
    plt.plot(times, total_counts, label="Total inventory")
    plt.title("Inventory Over Time")
    plt.xlabel("Time")
    plt.ylabel("Inventory Count")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    initial_inventory = np.loadtxt("data/initial_inventories/schedule_0_init_inventory.txt", delimiter=" ")
    schedule = np.loadtxt("data/schedules/schedule_0.txt", delimiter=" ")
    visualize_inventory(initial_inventory, schedule)
