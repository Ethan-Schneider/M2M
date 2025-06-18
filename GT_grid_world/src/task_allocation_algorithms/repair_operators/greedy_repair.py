import numpy as np
from typing import List, Tuple, Dict
from ...agent import AgentLoader
from ...analysis.statistics import Stats
from ...graph import Graph
from ...utils import manhattan_distance

import time

def greedy_repair(S: Stats, G: Graph, cost_tensor: np.ndarray, cost_tensor_agent_start: np.ndarray, Rs: AgentLoader,
                 start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]],
                 idx_to_task_id: Dict[int, int], temp_allocations: List[Tuple[int, int, int, int]],
                 method: str = "manhattan") -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], float]:
    """
    Greedily repair a solution by iteratively assigning the minimum cost allocation.
    
    Args:
        S: Statistics object
        G: Graph object
        cost_tensor: 4D numpy array of costs
        cost_tensor_agent_start: 2D numpy array of costs
        Rs: AgentLoader containing all agents
        start_locs: List of start locations
        goal_locs: List of goal locations
        idx_to_task_id: Mapping from tensor indices to task IDs
        temp_allocations: List of current allocations (agent_idx, task_idx, start_idx, goal_idx)
        method: Cost calculation method ("manhattan" or "shortest_path")
        
    Returns:
        Tuple containing:
        - List of tuples (m, n, p, q) representing allocations where:
          - m is the agent index
          - n is the task index
          - p is the start location index
          - q is the goal location index
        - Total cost of all allocations
    """
    total_cost = 0.0
    allocations = []
    M, __, P, __ = cost_tensor.shape
    
    while True:
        C = cost_tensor + cost_tensor_agent_start.reshape(M, 1, P, 1)

        # Get indices of minimum cost element
        m, n, p, q = np.unravel_index(np.argmin(C), C.shape)

        if C[m, n, p, q] == np.inf:
            break

        total_cost += C[m, n, p, q]
        
        # Add allocation to solution
        Rs.agents[m].task_sequence.append((idx_to_task_id[int(n)], start_locs[p], goal_locs[q]))
        allocations.append((m, n, p, q))

        total_cost += C[m, n, p, q]
        
        # Update statistics
        S.append_early_task_ids(idx_to_task_id[int(n)])
        S.add_actual_distance(idx_to_task_id[int(n)])
        S.add_actual_pickup_distance(idx_to_task_id[int(n)])
        S.add_actual_duration(idx_to_task_id[int(n)])
        S.add_actual_pickup_duration(idx_to_task_id[int(n)])
        
        tik = time.time()
        # Update tensor by setting inf for:
        # 1. All allocations for this task n
        cost_tensor[:, n, :, :] = np.inf
        # 2. All allocations using this start location p
        cost_tensor[:, :, p, :] = np.inf
        # 3. All allocations using this goal location q
        cost_tensor[:, :, :, q] = np.inf

        # 4. Update costs for agent-start allocation for agent m
        for p in range(P):
            if method == "manhattan":
                cost = manhattan_distance(Rs.agents[m].state, start_locs[p])
            elif method == "shortest_path":
                cost = G.get_distance(Rs.agents[m].state, start_locs[p])
            else:
                raise ValueError(f"Invalid cost calculation method: {method}")
            cost_tensor_agent_start[m, p] = cost

    return Rs, allocations, total_cost
