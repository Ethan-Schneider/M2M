import numpy as np
import time
from typing import Set, Tuple, List, Dict
from ...graph import Graph
from ...agent import AgentLoader
from ...analysis.statistics import Stats
from .construct_cost_tensor_pair import construct_cost_tensor_pair, manhattan_distance

def randomized_greedy_allocation(S : Stats, G : Graph, cost_tensor: np.ndarray, cost_tensor_agent_start: np.ndarray, Rs : AgentLoader, start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]], idx_to_task_id: Dict[int, int], method : str = "manhattan") -> Tuple[List[Tuple[int, int, int, int]], float]:
    """
    Perform randomized greedy allocation of tasks to agents based on k smallest cost elements in the tensor.
    
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

    k = 10  # Number of smallest elements to consider

    M, __, P, __ = cost_tensor.shape
    allocations = []
    total_cost = 0.0
    
    while True:
        # Add cost_tensor_agent_start to cost_tensor
        C = cost_tensor + cost_tensor_agent_start.reshape(M, 1, P, 1)

        # Get the k smallest elements and their indices
        flat_indices = np.argpartition(C.flatten(), k)[:k]
        k_smallest_costs = C.flatten()[flat_indices]
        
        # Randomly select one of the k smallest elements
        selected_idx = np.random.randint(0, k)
        selected_flat_idx = flat_indices[selected_idx]
        
        # Convert flat index back to 4D indices
        m, n, p, q = np.unravel_index(selected_flat_idx, C.shape)
        selected_cost = k_smallest_costs[selected_idx]

        if selected_cost == np.inf:
            break
        
        allocations.append((int(m), int(n), int(p), int(q)))

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

        total_cost += selected_cost
        
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

def randomized_greedy_call(S: Stats, G: Graph, Rs: AgentLoader, J: Set[Tuple], 
            strategy: str = "lns", map_name: str = None, t: int = 0, method : str = "manhattan") -> AgentLoader:
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
        return Rs, [], 0.0
    
    # Unassign tasks not currently being worked on by any agent
    for agent in Rs.agents:
        print(f"Agent {agent.id} task sequence: {agent.task_sequence}")
        while len(agent.task_sequence) > 1:
            agent.task_sequence.pop(-1)
        print(f"Agent {agent.id} task sequence: {agent.task_sequence}")

    # Construct cost tensor
    tik = time.time()
    cost_tensor, cost_tensor_agent_start, start_locs, goal_locs, idx_to_task_id = construct_cost_tensor_pair(J, Rs, G, method)
    tok = time.time()
    print(f"Time taken to construct cost tensor: {tok - tik} seconds")

    print(f"idx_to_task_id: {idx_to_task_id}")
    
    tik = time.time()
    allocations, total_cost = randomized_greedy_allocation(S, G, cost_tensor, cost_tensor_agent_start, Rs, start_locs, goal_locs, idx_to_task_id, method)
    tok = time.time()
    print(f"Time taken to perform randomized greedy allocation: {tok - tik} seconds")
    print(f"Allocations: {allocations}")
    print(f"Total cost: {total_cost}")
    
    return Rs, allocations, total_cost
