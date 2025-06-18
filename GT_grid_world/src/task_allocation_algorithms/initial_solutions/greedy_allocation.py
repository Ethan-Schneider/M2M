import numpy as np
import time
from typing import Set, Tuple, List, Dict
from ...graph import Graph
from ...agent import AgentLoader
from ...analysis.statistics import Stats
from .construct_cost_tensor import construct_cost_tensor, manhattan_distance

def greedy_allocation(S : Stats, G : Graph, cost_tensor: np.ndarray, cost_tensor_agent_start: np.ndarray, Rs : AgentLoader, start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]], idx_to_task_id: Dict[int, int], method : str = "manhattan") -> Tuple[List[Tuple[int, int, int, int]], float]:
    """
    Perform greedy allocation of tasks to agents based on minimum cost elements in the tensor.
    Args:
        S: Statistics object
        G: Graph object
        cost_tensor: 4D numpy array of shape (M, N, P, Q) containing costs
        cost_tensor_agent_start: 2D numpy array of shape (M, P) containing costs for agent-start allocation
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
    M, __, P, __ = cost_tensor.shape
    allocations = []
    total_cost = 0.0

    print(f"Cost tensor shape: {cost_tensor.shape}")
    
    while True:
        # Add cost_tensor_agent_start to cost_tensor
        C = cost_tensor + cost_tensor_agent_start.reshape(M, 1, P, 1)

        # Get indices of minimum cost element
        m, n, p, q = np.unravel_index(np.argmin(C), C.shape)

        if C[m, n, p, q] == np.inf:
            break

        allocations.append((int(m), int(n), int(p), int(q)))

        total_cost += C[m, n, p, q]

        # Update statistics
        S.append_early_task_ids(idx_to_task_id[int(n)])

        S.add_actual_distance(idx_to_task_id[int(n)])
        S.add_actual_pickup_distance(idx_to_task_id[int(n)])
        S.add_actual_duration(idx_to_task_id[int(n)])
        S.add_actual_pickup_duration(idx_to_task_id[int(n)])

        # Update agent's task sequence with (task_id, start_location_tuple, goal_location_tuple)
        Rs.agents[m].task_sequence.append((idx_to_task_id[int(n)], start_locs[p], goal_locs[q]))

        # Update agent status
        if Rs.agents[m].status == 0:
            Rs.agents[m].status = 1
        
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

def greedy_call(S: Stats, G: Graph, Rs: AgentLoader, J: Set[Tuple], method : str = "manhattan") -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], float]:
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
        List of allocations
        Total cost of all allocations
    """
    tik = time.time()

    num_allocated_tasks = 0
    for agent in Rs.agents:
        num_allocated_tasks += len(agent.task_sequence)
    if len(J) == num_allocated_tasks:  # No tasks to assign
        return Rs, [], 0.0
    
    # Unassign tasks not currently being worked on by any agent
    for agent in Rs.agents:
        while len(agent.task_sequence) > 1:
            agent.task_sequence.pop(-1)

    # Construct cost tensor
    cost_tensor, cost_tensor_agent_start, start_locs, goal_locs, idx_to_task_id = construct_cost_tensor(J, Rs, G, method)

    allocations, total_cost = greedy_allocation(S, G, cost_tensor, cost_tensor_agent_start, Rs, start_locs, goal_locs, idx_to_task_id, method)

    print(f"Time taken to perform greedy allocation: {time.time() - tik} seconds")
    print(f"Total cost: {total_cost}")

    return Rs, allocations, total_cost
