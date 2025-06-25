import numpy as np
import time
from typing import List, Tuple, Dict
from ...agent import AgentLoader
from ...analysis.statistics import Stats
from ...graph import Graph
from ...utils import manhattan_distance

def fast_SCF_repair(S: Stats, G: Graph, agent_start_cost_tensor: np.ndarray, start_goal_dist: np.ndarray, task_start_mask: np.ndarray, task_goal_mask: np.ndarray, Rs: AgentLoader,
                 start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]],
                 idx_to_task_id: Dict[int, int], temp_allocations: List[Tuple[int, int, int, int]],
                 method: str = "manhattan", cost_lookup: Dict[Tuple[int, int, int, int], int] = None) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], float]:
    """
    Repair a solution using Second Coordinate Fixing (SCF) algorithm.
    Iterates over tasks and assigns each task to the best available agent.
    
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
        cost_lookup: Dictionary to store allocation costs
        
    Returns:
        Tuple containing:
        - AgentLoader with updated task sequences
        - List of allocations (m, n, p, q)
        - Total cost of all allocations
    """
    total_cost = 0.0
    allocations = []
    M = len(Rs.agents)
    N = len(idx_to_task_id)
    P = len(start_locs)

    # Copy masks so we can update them
    task_start_mask_ = task_start_mask.copy()
    task_goal_mask_ = task_goal_mask.copy()
    assigned_tasks = set()

    # Iterate over all tasks (SCF approach)
    for n in range(N):
        if n in assigned_tasks:
            continue
            
        valid_p = np.where(task_start_mask_[n] == 1)[0]
        valid_q = np.where(task_goal_mask_[n] == 1)[0]
        if len(valid_p) == 0 or len(valid_q) == 0:
            continue

        # Find the best agent for this task (SCF: iterate over agents for each task)
        min_cost = np.inf
        best = None
        
        for m in range(M):
            # Vectorized cost computation for all valid (p, q) pairs
            total_costs = agent_start_cost_tensor[m, valid_p][:, None] + start_goal_dist[np.ix_(valid_p, valid_q)]

            min_idx = np.argmin(total_costs)
            min_cost_m = total_costs.flat[min_idx]
            if min_cost_m < min_cost:
                min_cost = min_cost_m
                p_idx, q_idx = np.unravel_index(min_idx, total_costs.shape)
                best = (m, valid_p[p_idx], valid_q[q_idx])
                
        if best is None or min_cost == np.inf:
            continue

        # Add the task to the allocation
        m, p, q = best
        allocations.append((int(m), idx_to_task_id[int(n)], int(p), int(q)))
        total_cost += min_cost
        assigned_tasks.add(n)

        # Store cost in lookup table if provided
        if cost_lookup is not None:
            cost_lookup[(int(m), idx_to_task_id[int(n)], int(p), int(q))] = int(min_cost)

        # Update statistics
        S.append_early_task_ids(idx_to_task_id[int(n)])
        S.add_actual_distance(idx_to_task_id[int(n)])
        S.add_actual_pickup_distance(idx_to_task_id[int(n)])
        S.add_actual_duration(idx_to_task_id[int(n)])
        S.add_actual_pickup_duration(idx_to_task_id[int(n)])

        # Update agent's task sequence
        Rs.agents[m].task_sequence.append((idx_to_task_id[int(n)], start_locs[p], goal_locs[q]))
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
                cost = manhattan_distance(goal_locs[q], start_locs[p_])
            elif method == "shortest_path":
                cost = G.get_distance(goal_locs[q], start_locs[p_])
            else:
                raise ValueError(f"Invalid cost calculation method: {method}")
            agent_start_cost_tensor[m, p_] = cost

    return Rs, allocations, total_cost
