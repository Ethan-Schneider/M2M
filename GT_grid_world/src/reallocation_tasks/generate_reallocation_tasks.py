import numpy as np

from ..agent import AgentLoader
from ..graph import Graph

# schedule_generation writes 0-indexed SKU ids; GT warehouse SKUs are 1-indexed.
SCHEDULE_SKU_ID_OFFSET = 1
TASK_TYPE_SHUFFLE = 2


def generate_reallocation_tasks(schedule: np.ndarray, J: dict, G : Graph, Rs: AgentLoader, B : int, W : int, t : int) -> dict:
    """
    Look at future released tasks between timesteps [t, t+W], find the subset of tasks that we want to reallocate for then construct a set of reallocation tasks to return.

    Args:
        schedule: The schedule of tasks to be executed.
        J: The dictionary of tasks.
        G: The graph of the warehouse.
        Rs: The dictionary of agents.
        B: The beginning of timestep window.
        W: The end of timestep window.

    Returns:
        A dictionary of reallocation tasks.
        
        Reallocation task is defined as: tau=(C_i, r_i, d_i, sigma_i), where
        C_i = {(S_i^k, D_i^k)}_{k=1}^{|C_i|} is the set of (start_loc, end_loc) pairs for task i.
        r_i, d_i are real numbers for the release time and deadline of the reallocation task
        sigma_i is the task type, 0 for outbound, 1 for inbound, 2 for shuffle.
    """

    # Get set of flagged real tasks: T_{flag}^r = {tau_j^r \in T^r : t+B <= r_j <= t+W and sigma_j = 0}
    # Lookahead at the schedule to find the future tasks
    flagged_real_tasks = [tau for tau in schedule if t + B <= tau[0] <= t + W and tau[3] == 0]

    reallocation_tasks = {}
    aisle_columns = sorted({loc[1] for loc in G.get_aisle_locations()})

    for reallocation_task_id, task in enumerate(flagged_real_tasks):
        sku_id = int(task[2]) + SCHEDULE_SKU_ID_OFFSET
        sku_locations = set(G.warehouse.get_sku_instances(sku_id))
        empty_locations = set(G.warehouse.get_empty_locations())

        C_i = set()
        for column in aisle_columns:
            aisle_locations = [
                loc for loc in G.get_aisle_locations() if loc[1] == column
            ]
            start_locations = [
                loc for loc in aisle_locations if loc in sku_locations
            ]
            goal_locations = [
                loc for loc in aisle_locations if loc in empty_locations
            ]
            # If there are no start and or goal locations, skip this tuple
            if not start_locations or not goal_locations:
                continue
            for start_loc in start_locations:
                for goal_loc in goal_locations:
                    C_i.add((start_loc, goal_loc))

        # If C_i is empty ,skip this task
        if not C_i:
            continue

        r_i = t
        d_i = int(task[0])
        sigma_i = TASK_TYPE_SHUFFLE
        reallocation_tasks[reallocation_task_id] = (C_i, r_i, d_i, sigma_i)

    return reallocation_tasks