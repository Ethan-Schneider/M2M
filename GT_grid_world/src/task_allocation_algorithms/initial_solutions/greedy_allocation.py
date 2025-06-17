import numpy as np
import time
from typing import Set, Tuple, List, Dict
from ...graph import Graph
from ...agent import AgentLoader
from ...analysis.statistics import Stats
from .construct_cost_tensor import construct_cost_tensor, manhattan_distance

def greedy_allocation(S, cost_tensor: np.ndarray, Rs : AgentLoader, start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]], idx_to_task_id: Dict[int, int], method : str = "manhattan") -> Tuple[List[Tuple[int, int, int, int]], float]:
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
    
    while True:
        # Find minimum cost element
        min_cost = np.min(working_tensor)
        if min_cost == np.inf:
            break
            
        # Get indices of minimum cost element
        m, n, p, q = np.unravel_index(np.argmin(working_tensor), working_tensor.shape)
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

                                        if method == "manhattan":
                                        # Calculate cost from current goal to new start
                                            goal_to_start_cost = manhattan_distance(current_goal_loc, new_start_loc)
                                            # Calculate cost from new start to new goal
                                            start_to_goal_cost = manhattan_distance(new_start_loc, new_goal_loc)
                                            working_tensor[m, n, p, q] = goal_to_start_cost + start_to_goal_cost
                                        elif method == "shortest_path":
                                            goal_to_start_cost = G.get_distance(current_goal_loc, new_start_loc)
                                            start_to_goal_cost = G.get_distance(new_start_loc, new_goal_loc)
                                            working_tensor[m, n, p, q] = goal_to_start_cost + start_to_goal_cost
                                        else:
                                            raise ValueError(f"Invalid cost calculation method: {method}")

    return allocations, total_cost

def greedy_call(S: Stats, G: Graph, Rs: AgentLoader, J: Set[Tuple], 
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
        return Rs
    
    # Unassign tasks not currently being worked on by any agent
    for agent in Rs.agents:
        print(f"Agent {agent.id} task sequence: {agent.task_sequence}")
        while len(agent.task_sequence) > 1:
            agent.task_sequence.pop(-1)
        print(f"Agent {agent.id} task sequence: {agent.task_sequence}")

    # Construct cost tensor
    tik = time.time()
    cost_tensor, start_locs, goal_locs, idx_to_task_id = construct_cost_tensor(J, Rs, G, method)
    tok = time.time()
    print(f"Time taken to construct cost tensor: {tok - tik} seconds")

    print(f"idx_to_task_id: {idx_to_task_id}")
    
    tik = time.time()
    allocations, total_cost = greedy_allocation(S, cost_tensor, Rs, start_locs, goal_locs, idx_to_task_id, method)
    tok = time.time()
    print(f"Time taken to perform greedy allocation: {tok - tik} seconds")
    print(f"Allocations: {allocations}")
    print(f"Total cost: {total_cost}")
    print(f"Robot Task Sequences")
    for agent in Rs.agents:
        print(f"Agent {agent.id} task sequence: {agent.task_sequence}")
    
    return Rs
