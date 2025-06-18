import numpy as np
from typing import List, Tuple, Dict
from ...agent import AgentLoader
from ...analysis.statistics import Stats
from ...graph import Graph
from ...utils import manhattan_distance

import time

def greedy_repair(S: Stats, G: Graph, cost_tensor: np.ndarray, solution: AgentLoader,
                 start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]],
                 idx_to_task_id: Dict[int, int], temp_allocations: List[Tuple[int, int, int, int]],
                 method: str = "manhattan") -> AgentLoader:
    """
    Greedily repair a solution by iteratively assigning the minimum cost allocation.
    
    Args:
        S: Statistics object
        G: Graph object
        cost_tensor: 4D numpy array of costs
        solution: Current solution to repair
        start_locs: List of start locations
        goal_locs: List of goal locations
        idx_to_task_id: Mapping from tensor indices to task IDs
        temp_allocations: List of current allocations (agent_idx, task_idx, start_idx, goal_idx)
        method: Cost calculation method ("manhattan" or "shortest_path")
        
    Returns:
        Repaired solution
    """
    greedy_repair_time = 0.0
    tik = time.time()
    M, N, P, Q = cost_tensor.shape
    working_tensor = cost_tensor.copy()
    
    # Track each agent's current goal location
    agent_goal_locs = [None] * M
    for agent_idx, agent in enumerate(solution.agents):
        if agent.task_sequence:
            last_task = agent.task_sequence[-1]
            goal_loc = last_task[2]  # goal location of last task
            # Find index in goal_locs
            for i, loc in enumerate(goal_locs):
                if loc == goal_loc:
                    agent_goal_locs[agent_idx] = i
                    break
    
    while True:
        # Check if there are any valid allocations left
        if np.all(np.isinf(working_tensor)):
            break
            
        # Get indices of minimum cost element
        m, n, p, q = np.unravel_index(np.argmin(working_tensor), working_tensor.shape)
        
        # Add allocation to solution
        task_id = idx_to_task_id[int(n)]
        solution.agents[m].task_sequence.append((task_id, start_locs[p], goal_locs[q]))
        
        # Update statistics
        S.append_early_task_ids(task_id)
        S.add_actual_distance(task_id)
        S.add_actual_pickup_distance(task_id)
        S.add_actual_duration(task_id)
        S.add_actual_pickup_duration(task_id)
        
        # Update agent's goal location
        agent_goal_locs[m] = q
        
        tik = time.time()
        # Update tensor by setting inf for:
        # 1. All allocations for this task n
        working_tensor[:, n, :, :] = np.inf
        # 2. All allocations using this start location p
        working_tensor[:, :, p, :] = np.inf
        # 3. All allocations using this goal location q
        working_tensor[:, :, :, q] = np.inf
        
        # Add new allocation to temp_allocations
        temp_allocations.append((m, n, p, q))
        
        # 4. Update costs for all remaining allocations for this agent
        # For each remaining task and start location
        for n in range(N):
            if working_tensor[m, n, :, :].min() != np.inf:
                for p in range(P):
                    if working_tensor[m, n, p, :].min() != np.inf:
                        for q in range(Q):
                            if working_tensor[m, n, p, q] != np.inf:
                                # Calculate new cost from agent's current goal location to new start location
                                if agent_goal_locs[m] is not None:
                                    # Get the actual locations
                                    current_goal_loc = goal_locs[agent_goal_locs[m]]
                                    new_start_loc = start_locs[p]
                                    new_goal_loc = goal_locs[q]
                                    # Calculate cost from current goal to new start
                                    if method == "manhattan":
                                        goal_to_start_cost = manhattan_distance(current_goal_loc, new_start_loc)
                                        # Calculate cost from new start to new goal
                                        start_to_goal_cost = manhattan_distance(new_start_loc, new_goal_loc)
                                    else:  # shortest_path
                                        goal_to_start_cost = G.get_distance(current_goal_loc, new_start_loc)
                                        start_to_goal_cost = G.get_distance(new_start_loc, new_goal_loc)
                                    working_tensor[m, n, p, q] = goal_to_start_cost + start_to_goal_cost
        tok = time.time()
        greedy_repair_time += tok - tik
        print(f"Greedy repair time: {tok-tik}")

    print(f"TotalGreedy repair time: {greedy_repair_time}")
    return solution
