import numpy as np
from typing import List, Tuple, Dict, Set
from ...agent import AgentLoader
from ...analysis.statistics import Stats
from ...graph import Graph
from ...utils import manhattan_distance

import time

def greedy_repair(S: Stats, G: Graph, agent_start_cost_tensor: np.ndarray, start_goal_dist: np.ndarray, task_start_mask: np.ndarray, task_goal_mask: np.ndarray, Rs: AgentLoader,
                 start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]],
                 idx_to_task_id: Dict[int, int], temp_allocations: List[Tuple[int, int, int, int]],
                 method: str = "manhattan", J: Set[Tuple] = None, cost_lookup: Dict[Tuple[int, int, int, int], int] = None) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], float]:
    """
    Greedily repair a solution by iteratively assigning the minimum cost allocation using cost elements.
    Args:
        S: Statistics object
        G: Graph object
        agent_start_cost_tensor: (M, P) array
        start_goal_dist: (P, Q) array
        task_start_mask: (N, P) array
        task_goal_mask: (N, Q) array
        Rs: AgentLoader containing all agents
        start_locs: List of start locations
        goal_locs: List of goal locations
        idx_to_task_id: Mapping from tensor indices to task IDs
        temp_allocations: List of current allocations (agent_idx, task_idx, start_idx, goal_idx)
        method: Cost calculation method ("manhattan" or "shortest_path")
    Returns:
        Tuple containing:
        - AgentLoader with updated task sequences
        - List of allocations (m, n, p, q)
        - Total cost of all allocations
    """
    total_cost = 0.0
    allocations = temp_allocations.copy()
    M = len(Rs.agents)
    N = len(idx_to_task_id)
    P = len(start_locs)

    # Copy masks so we can update them
    task_start_mask_ = task_start_mask.copy()
    task_goal_mask_ = task_goal_mask.copy()
    assigned_tasks = set()

    while True:
        # If all tasks are assigned, if no start or goal locations are left, break
        if len(assigned_tasks) == N:
            break
        if np.all(task_start_mask_ == 0) or np.all(task_goal_mask_ == 0):
            break

        best_cost = -np.inf
        best = None
        # For each task, find the best (m, n, p, q)
        for n in range(N):
            # If the task is already assigned, skip
            if n in assigned_tasks:
                continue

            # Find all valid start and goal locations for the task
            valid_p = np.where(task_start_mask_[n] == 1)[0]
            valid_q = np.where(task_goal_mask_[n] == 1)[0]

            # If there are no valid start or goal locations, skip
            if len(valid_p) == 0 or len(valid_q) == 0:
                continue

            # Compute the total cost for all valid (p, q) pairs for the task
            agent_costs = agent_start_cost_tensor[:, valid_p]  # (M, len(valid_p))
            sg_costs = start_goal_dist[np.ix_(valid_p, valid_q)]  # (len(valid_p), len(valid_q))
            total_costs = agent_costs[:, :, None] + sg_costs[None, :, :]

            # Find the index of the maximum cost (since costs are negative, this minimizes distance)
            min_idx = np.argmax(total_costs)
            new_cost = total_costs.flat[min_idx]

            # If the new cost is greater than the current best cost, update the best allocation
            if new_cost > best_cost:
                # If there are multiple maximum costs, choose one randomly
                if np.sum(total_costs == new_cost) > 1:
                    max_locations = np.where(total_costs == new_cost)
                    max_locations_list = [(int(max_locations[0][i]), int(max_locations[1][i]), int(max_locations[2][i])) for i in range(len(max_locations[0]))]

                    random_idx = np.random.choice(range(len(max_locations_list)), 1)[0]
                    m_idx, p_idx, q_idx = max_locations_list[random_idx]

                    best_cost = new_cost
                    best = (m_idx, n, valid_p[p_idx], valid_q[q_idx])
                else:
                    best_cost = new_cost
                    m_idx, p_idx, q_idx = np.unravel_index(min_idx, total_costs.shape)
                    best = (m_idx, n, valid_p[p_idx], valid_q[q_idx])

        if best is None or best_cost == -np.inf:
            break

        # Add the best task to the allocation
        m, n, p, q = best
        allocations.append((int(m), idx_to_task_id[int(n)], int(p), int(q)))

        # If cost lookup is provided, store the cost in the lookup table
        if cost_lookup is not None:
            cost_lookup[(int(m), idx_to_task_id[int(n)], int(p), int(q))] = int(best_cost)
        total_cost += best_cost
        assigned_tasks.add(n)

        # Update statistics
        S.append_early_task_ids(idx_to_task_id[int(n)])
        S.add_actual_distance(idx_to_task_id[int(n)])
        S.add_actual_pickup_distance(idx_to_task_id[int(n)])
        S.add_actual_duration(idx_to_task_id[int(n)])
        S.add_actual_pickup_duration(idx_to_task_id[int(n)])

        # Update agent's task sequence
        deadline = next(task[3] for task in J if task[0] == idx_to_task_id[int(n)])
        Rs.agents[m].task_sequence.append((idx_to_task_id[int(n)], start_locs[p], goal_locs[q], deadline))
        if Rs.agents[m].status == 0:
            Rs.agents[m].status = 1

        # Invalidate this task, start, and goal for all future agents
        task_start_mask_[:, p] = 0
        task_goal_mask_[:, q] = 0
        task_start_mask_[n, :] = 0
        task_goal_mask_[n, :] = 0

        # Update costs for agent-start allocation for agent m
        for p_ in range(P):
            if method == "manhattan":
                cost = -1.0 * manhattan_distance(goal_locs[q], start_locs[p_])
            elif method == "shortest_path":
                cost = -1.0 * G.get_distance(goal_locs[q], start_locs[p_])
            else:
                raise ValueError(f"Invalid cost calculation method: {method}")
            agent_start_cost_tensor[m, p_] = cost

    return Rs, allocations, total_cost, cost_lookup
