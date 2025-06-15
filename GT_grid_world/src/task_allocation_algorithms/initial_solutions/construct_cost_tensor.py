import numpy as np
from typing import Set, Tuple
from ...graph import Graph
from ...agent import AgentLoader

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
    
    # Calculate costs only for valid combinations
    for m in range(M):
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