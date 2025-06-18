import numpy as np
import time
from typing import Set, Tuple, List, Dict
from ...graph import Graph
from ...agent import AgentLoader
from ...analysis.statistics import Stats
from .construct_cost_tensor import construct_cost_tensor

def random_allocation(S : Stats, cost_tensor: np.ndarray, cost_tensor_agent_start: np.ndarray, Rs : AgentLoader, start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]], idx_to_task_id: Dict[int, int], method : str = "manhattan") -> Tuple[List[Tuple[int, int, int, int]], float]:
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
    M, __, P, __ = cost_tensor.shape
    allocations = []
    total_cost = 0.0

    C = cost_tensor + cost_tensor_agent_start.reshape(M, 1, P, 1)
    
    while True:
        # Get valid (non-infinite) entries
        valid_entries = np.where(C != np.inf)
        if len(valid_entries[0]) == 0:
            break
            
        # Randomly select one of the valid entries
        random_idx = np.random.randint(0, len(valid_entries[0]))
        m = valid_entries[0][random_idx]
        n = valid_entries[1][random_idx]
        p = valid_entries[2][random_idx]
        q = valid_entries[3][random_idx]

        total_cost += C[m, n, p, q]
        
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
        
        # Update tensor by setting inf for:
        # 1. All allocations for this task n
        C[:, n, :, :] = np.inf
        # 2. All allocations using this start location p
        C[:, :, p, :] = np.inf
        # 3. All allocations using this goal location q
        C[:, :, :, q] = np.inf

    return allocations, total_cost

def random_call(S: Stats, G: Graph, Rs: AgentLoader, J: Set[Tuple], method : str = "manhattan") -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], float]:
    """
    Multi-Agent to Multi-Task Large Neighborhood Search algorithm.
    
    Args:
        S: Statistics object for tracking metrics
        G: Graph representing the warehouse
        Rs: AgentLoader containing all agents
        J: Set of tasks to be assigned
        
    Returns:
        Updated AgentLoader with assigned tasks
        List of allocations
        Total cost of all allocations
    """
    if not J:  # No tasks to assign
        return Rs, [], 0.0
    
    # Unassign tasks not currently being worked on by any agent
    for agent in Rs.agents:
        while len(agent.task_sequence) > 1:
            agent.task_sequence.pop(0)

    # Construct cost tensor
    tik = time.time()
    cost_tensor, cost_tensor_agent_start, start_locs, goal_locs, idx_to_task_id = construct_cost_tensor(J, Rs, G, method)
    tok = time.time()
    print(f"Time taken to construct cost tensor: {tok - tik} seconds")
    
    tik = time.time()
    allocations, total_cost = random_allocation(S, cost_tensor, cost_tensor_agent_start, Rs, start_locs, goal_locs, idx_to_task_id, method)
    tok = time.time()
    print(f"Time taken to perform greedy allocation: {tok - tik} seconds")
    print(f"Allocations: {allocations}")
    print(f"Total cost: {total_cost}")
    print(f"Robot Task Sequences")
    for agent in Rs.agents:
        print(f"Agent {agent.id} task sequence: {agent.task_sequence}")
    
    return Rs, allocations, total_cost
