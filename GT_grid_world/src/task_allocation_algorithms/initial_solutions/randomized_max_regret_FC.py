import numpy as np
import time
import random
from typing import Set, Tuple, List, Dict
from ...graph import Graph
from ...agent import AgentLoader
from ...analysis.statistics import Stats
from .construct_cost_tensor import construct_cost_tensor, manhattan_distance

def randomized_max_regret_FC_allocation(S : Stats, G : Graph, cost_tensor: np.ndarray, cost_tensor_agent_start: np.ndarray, Rs : AgentLoader, start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]], idx_to_task_id: Dict[int, int], method : str = "manhattan", top_percentage: float = 0.2) -> Tuple[List[Tuple[int, int, int, int]], float]:
    """
    Perform randomized max regret FC allocation of tasks to agents based on minimum cost elements in the tensor.
    
    Args:
        S: Statistics object for tracking metrics
        G: Graph representing the warehouse
        cost_tensor: 4D numpy array of shape (M, N, P, Q) containing costs
        Rs: AgentLoader containing all agents
        start_locs: List of start locations
        goal_locs: List of goal locations
        top_percentage: Percentage of top regrets to randomly select from (default 0.2 = 20%)
    Returns:
        Tuple containing:
        - List of tuples (m, n, p, q) representing allocations where:
          - m is the agent index
          - n is the task index
          - p is the start location index
          - q is the goal location index
        - Total cost of all allocations
    """
    M, N, P, __ = cost_tensor.shape
    allocations = []
    total_cost = 0.0
    
    unallocated_tasks = list(range(N))
    while True:
        # Calculate regret for each task and record all regrets
        regret_data = []  # List of (regret, indices) tuples
        
        for n in unallocated_tasks:
            C = cost_tensor[:, n, :, :] + cost_tensor_agent_start.reshape(M, P, 1)
                
            # Find the first and second minimum values of C
            flat_costs = C.flatten()
            valid_costs = flat_costs[flat_costs != np.inf]
            
            if len(valid_costs) < 2:  # Skip if there are fewer than 2 valid costs
                continue
            
            # Sort valid costs and get first and second minimum
            sorted_costs = np.sort(valid_costs)
            first_min = sorted_costs[0]
            second_min = sorted_costs[1]
            
            # Calculate regret (absolute difference between first and second min)
            regret = abs(second_min - first_min)
            
            # Get the indices of the minimum cost for this task
            min_idx_3d = np.unravel_index(np.argmin(C), C.shape)
            m = min_idx_3d[0]
            p = min_idx_3d[1]
            q = min_idx_3d[2]
            indices = (m, n, p, q)
            
            # Record regret and corresponding indices
            regret_data.append((regret, indices))
        
        # If no valid regrets found, break
        if not regret_data:
            break
            
        # Sort regrets in descending order
        regret_data.sort(key=lambda x: x[0], reverse=True)
        
        # Select top X% regrets
        num_top_regrets = max(1, int(len(regret_data) * top_percentage))
        top_regrets = regret_data[:num_top_regrets]
        
        # Randomly select one from the top regrets
        __, max_regret_indices = random.choice(top_regrets)
            
        # Use the selected indices
        m, n, p, q = max_regret_indices

        if n in unallocated_tasks:
            unallocated_tasks.remove(n)

        val = cost_tensor[m, n, p, q] + cost_tensor_agent_start[m, p]

        if val == np.inf:
            break

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

        total_cost += val

        if len(unallocated_tasks) == 0:
            break
        
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

    return allocations, total_cost

def randomized_max_regret_FC_call(S: Stats, G: Graph, Rs: AgentLoader, J: Set[Tuple], method : str = "manhattan", top_percentage: float = 0.2) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], float]:
    """
    Multi-Agent to Multi-Task Large Neighborhood Search algorithm.
    
    Args:
        S: Statistics object for tracking metrics
        G: Graph representing the warehouse
        Rs: AgentLoader containing all agents
        J: Set of tasks to be assigned
        method: Cost calculation method
        top_percentage: Percentage of top regrets to randomly select from (default 0.2 = 20%)
        
    Returns:
        Updated AgentLoader with assigned tasks
        List of allocations
        Total cost of all allocations
    """
    if not J:  # No tasks to assign
        return Rs, [], 0.0
    
    # Unassign tasks not currently being worked on by any agent
    for agent in Rs.agents:
        print(f"Agent {agent.id} task sequence: {agent.task_sequence}")
        while len(agent.task_sequence) > 1:
            agent.task_sequence.pop(-1)
        print(f"Agent {agent.id} task sequence: {agent.task_sequence}")

    # Construct cost tensor
    tik = time.time()
    cost_tensor, cost_tensor_agent_start, start_locs, goal_locs, idx_to_task_id = construct_cost_tensor(J, Rs, G, method)
    tok = time.time()
    print(f"Time taken to construct cost tensor: {tok - tik} seconds")

    print(f"idx_to_task_id: {idx_to_task_id}")
    
    tik = time.time()
    allocations, total_cost = randomized_max_regret_FC_allocation(S, G, cost_tensor, cost_tensor_agent_start, Rs, start_locs, goal_locs, idx_to_task_id, method, top_percentage)
    tok = time.time()
    print(f"Time taken to perform randomized max regret FC allocation: {tok - tik} seconds")
    
    return Rs, allocations, total_cost
