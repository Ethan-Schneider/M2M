import numpy as np
import time
from typing import Set, Tuple, List
from ..graph import Graph
from ..agent import AgentLoader
from ..analysis.statistics import Stats
from ..utils import get_task_start_location, get_task_goal_location

def manhattan_distance(loc1: Tuple[int, int], loc2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two locations."""
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

def construct_cost_tensor(J: Set[Tuple], Rs: AgentLoader, G: Graph) -> np.ndarray:
    """
    Construct a 4D cost tensor for task allocation.
    
    Args:
        J: Set of tasks, where each task is (task_id, start_locations_frozenset, goal_locations_frozenset)
        Rs: AgentLoader containing all agents
        G: Graph representing the warehouse
        
    Returns:
        4D numpy array of shape (M, N, P, Q) where:
        - M is number of agents
        - N is number of tasks
        - P is number of possible start locations
        - Q is number of possible goal locations
    """
    M = len(Rs.agents)  # Number of agents
    N = len(J)  # Number of tasks

    # Get all possible start and goal locations from tasks
    all_start_locs = set()
    all_goal_locs = set()
    for task in J:
        all_start_locs.update(task[1])  # start_locations_frozenset
        all_goal_locs.update(task[2])  # goal_locations_frozenset

    allocated_tasks = set()
    for agent in Rs.agents:
        for task in agent.task_sequence:
            allocated_tasks.add(task[0])
    
    # Convert to sorted lists for consistent indexing
    start_locs = sorted(list(all_start_locs))
    goal_locs = sorted(list(all_goal_locs))
    P = len(start_locs)
    Q = len(goal_locs)
    
    # Create location to index mappings
    start_loc_to_idx = {loc: idx for idx, loc in enumerate(start_locs)}
    goal_loc_to_idx = {loc: idx for idx, loc in enumerate(goal_locs)}

    # Initialize cost tensor with infinity
    cost_tensor = np.full((M, N, P, Q), np.inf)
    
    # Get all locations currently allocated to tasks
    allocated_locs = set()
    for agent in Rs.agents:
        for task in agent.task_sequence:
            allocated_locs.update(task[1])
            allocated_locs.update(task[2])
    
    # Get each agent's final location
    agent_final_locs = []
    for agent in Rs.agents:
        if agent.task_sequence:
            # Find the last task in the sequence
            last_task_id = agent.task_sequence[-1]
            agent_final_locs.append(get_task_goal_location(J, last_task_id))
        else:
            # If agent has no tasks, use current state
            agent_final_locs.append(agent.state)
    
    # Calculate costs only for valid combinations
    for m, agent_final_loc in enumerate(agent_final_locs):
        for n, task in enumerate(J):
            if n in allocated_tasks:
                continue
            # Get valid start and goal locations for this task
            valid_starts = [loc for loc in task[1] if loc not in allocated_locs]
            valid_goals = [loc for loc in task[2] if loc not in allocated_locs]
            
            # Calculate costs for valid combinations
            for start_loc in valid_starts:
                i = start_loc_to_idx[start_loc]
                for goal_loc in valid_goals:
                    j = goal_loc_to_idx[goal_loc]
                    # Cost is sum of:
                    # 1. Distance from agent's final location to start location
                    # 2. Distance from start location to goal location

                    cost = (manhattan_distance(Rs.agents[m].state, start_loc) + 
                           manhattan_distance(start_loc, goal_loc))
                    
                    cost_tensor[m, n, i, j] = cost
    
    return cost_tensor, start_locs, goal_locs

def greedy_allocation(S, cost_tensor: np.ndarray, Rs : AgentLoader, start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int, int, int]], float]:
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
        S.append_early_task_ids(int(n))

        S.add_actual_distance(int(n))
        S.add_actual_pickup_distance(int(n))
        S.add_actual_duration(int(n))
        S.add_actual_pickup_duration(int(n))

        # Update agent's task sequence with (task_id, start_location_tuple, goal_location_tuple)
        Rs.agents[m].task_sequence.append((int(n), start_locs[p], goal_locs[q]))

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

def M2M_lns(S: Stats, G: Graph, Rs: AgentLoader, J: Set[Tuple], 
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
            agent.task_sequence.pop(0)

    # Construct cost tensor
    tik = time.time()
    cost_tensor, start_locs, goal_locs = construct_cost_tensor(J, Rs, G)
    tok = time.time()
    print(f"Time taken to construct cost tensor: {tok - tik} seconds")
    
    tik = time.time()
    allocations, total_cost = greedy_allocation(S, cost_tensor, Rs, start_locs, goal_locs)
    tok = time.time()
    print(f"Time taken to perform greedy allocation: {tok - tik} seconds")
    print(f"Allocations: {allocations}")
    print(f"Total cost: {total_cost}")
    print(f"Robot Task Sequences")
    for agent in Rs.agents:
        print(f"Agent {agent.id} task sequence: {agent.task_sequence}")
    
    return Rs
