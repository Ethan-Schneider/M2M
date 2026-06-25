import numpy as np
import matplotlib.pyplot as plt

QUEUE_SKU_ID = 0
QUEUE_TASK_TYPE = 1
TASK_TYPE_OUTBOUND = 0
TASK_TYPE_INBOUND = 1


def _num_skus(initial_inventory: np.ndarray, queue: np.ndarray) -> int:
    initial_inventory = np.atleast_2d(initial_inventory)
    queue = np.atleast_2d(queue)
    max_sku_id = -1
    if initial_inventory.size:
        max_sku_id = max(max_sku_id, int(initial_inventory[:, 0].max()))
    if queue.size:
        max_sku_id = max(max_sku_id, int(queue[:, QUEUE_SKU_ID].max()))
    return max_sku_id + 1


def compute_inventory_tally(initial_counts: dict, queue: np.ndarray) -> dict:
    counts = dict(initial_counts)
    for sku_id, task_type in queue.astype(int):
        if task_type == TASK_TYPE_OUTBOUND:
            counts[sku_id] -= 1
        elif task_type == TASK_TYPE_INBOUND:
            counts[sku_id] += 1
    return counts


def visualize_queue(initial_inventory: np.ndarray, queue: np.ndarray) -> None:
    initial_inventory = np.atleast_2d(np.asarray(initial_inventory))
    queue = np.atleast_2d(np.asarray(queue)).astype(int)

    num_skus = _num_skus(initial_inventory, queue)
    initial_counts = {sku_id: 0 for sku_id in range(num_skus)}
    for sku_id, _, _ in initial_inventory:
        initial_counts[int(sku_id)] += 1

    positions = list(range(queue.shape[0] + 1))
    total_counts = []
    for prefix_len in positions:
        prefix = queue[:prefix_len]
        total_counts.append(sum(compute_inventory_tally(initial_counts, prefix).values()))

    plt.figure(figsize=(10, 5))
    plt.plot(positions, total_counts, label="Total projected inventory")
    plt.title("Projected Inventory Over Queue Prefix")
    plt.xlabel("Tasks popped from queue")
    plt.ylabel("Inventory Count")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    initial_inventory = np.loadtxt("data/initial_inventories/queue_0_init_inventory.txt", delimiter=" ")
    queue = np.loadtxt("data/queues/queue_0.txt", delimiter=" ")
    visualize_queue(initial_inventory, queue)
