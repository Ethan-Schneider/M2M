import numpy as np
import time
from typing import Set, Tuple, List, Dict
from ...graph import Graph
from ...agent import AgentLoader
from ...analysis.statistics import Stats
from .construct_cost_tensor import construct_cost_tensor, manhattan_distance

def FCF_allocation(S, cost_tensor: np.ndarray, Rs : AgentLoader, start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]], idx_to_task_id: Dict[int, int]) -> Tuple[List[Tuple[int, int, int, int]], float]:
    """
    Perform greedy allocation of tasks to agents based on minimum cost elements in the tensor.
    
    Args:
        cost_tensor: 4D numpy array of shape (M, N, P, Q) containing costs
        Rs: AgentLoader containing all agents
        start_locs: List of start locations
        goal_locs: List of goal locations
    Returns:
        Tuple containing:
        - List of tuples (m, n, p, q) representing allocations where:
          - m is the agent index
          - n is the task index
          - p is the start location index
          - q is the goal location index
        - Total cost of all allocations
    """
    M, N, P, Q = cost_tensor.shape
    allocations = []
    working_tensor = cost_tensor.copy()
    total_cost = 0.0
    
    # Track each agent's current goal location
    agent_goal_locs = [None] * M

    for n in range(N):
        # Find minimum cost element
        min_cost = np.min(working_tensor[:, n, :, :])
        if min_cost == np.inf:
            continue

        # Get the 3D slice for this task
        task_slice = working_tensor[:, n, :, :]
        # Find min in the 3D slice
        min_idx_3d = np.unravel_index(np.argmin(task_slice), task_slice.shape)
        # Convert back to 4D indices
        m = min_idx_3d[0]
        p = min_idx_3d[1]
        q = min_idx_3d[2]
        
        allocations.append((int(m), idx_to_task_id[int(n)], int(p), int(q)))

        # Update statistics
        S.append_early_task_ids(idx_to_task_id[int(n)])

        S.add_actual_distance(idx_to_task_id[int(n)])
        S.add_actual_pickup_distance(idx_to_task_id[int(n)])
        S.add_actual_duration(idx_to_task_id[int(n)])
        S.add_actual_pickup_duration(idx_to_task_id[int(n)])

        # Update agent's task sequence with (task_id, start_location_tuple, goal_location_tuple)
        Rs.agents[m].task_sequence.append((idx_to_task_id[int(n)], start_locs[p], goal_locs[q]))

        if Rs.agents[m].status == 0:
            Rs.agents[m].status = 1

        total_cost += min_cost
        
        # Update agent's goal location
        agent_goal_locs[m] = q
        
        # Update tensor by setting inf for:
        # 1. All allocations for this task n
        working_tensor[:, n, :, :] = np.inf
        # 2. All allocations using this start location p
        working_tensor[:, :, p, :] = np.inf
        # 3. All allocations using this goal location q
        working_tensor[:, :, :, q] = np.inf
        
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
                                        goal_to_start_cost = manhattan_distance(current_goal_loc, new_start_loc)
                                        # Calculate cost from new start to new goal
                                        start_to_goal_cost = manhattan_distance(new_start_loc, new_goal_loc)
                                        working_tensor[m, n, p, q] = goal_to_start_cost + start_to_goal_cost

    return allocations, total_cost

def FCF_call(S: Stats, G: Graph, Rs: AgentLoader, J: Set[Tuple], 
            strategy: str = "lns", map_name: str = None, t: int = 0) -> AgentLoader:
    """
    Multi-Agent to Multi-Task Large Neighborhood Search algorithm.
    
    Args:
        S: Statistics object for tracking metrics
        G: Graph representing the warehouse
        Rs: AgentLoader containing all agents
        J: Set of tasks to be assigned
        strategy: Assignment strategy (currently only "lns" supported)
        map_name: Name of the map being used
        t: Current timestep
        
    Returns:
        Updated AgentLoader with assigned tasks
    """
    if not J:  # No tasks to assign
        return Rs
    
    # Unassign tasks not currently being worked on by any agent
    for agent in Rs.agents:
        while len(agent.task_sequence) > 1:
            agent.task_sequence.pop(-1)

    # Construct cost tensor
    tik = time.time()
    cost_tensor, start_locs, goal_locs, idx_to_task_id = construct_cost_tensor(J, Rs, G)
    tok = time.time()
    print(f"Time taken to construct cost tensor: {tok - tik} seconds")
    
    tik = time.time()
    allocations, total_cost = FCF_allocation(S, cost_tensor, Rs, start_locs, goal_locs, idx_to_task_id)
    tok = time.time()
    print(f"Time taken to perform FCF allocation: {tok - tik} seconds")
    
    return Rs
