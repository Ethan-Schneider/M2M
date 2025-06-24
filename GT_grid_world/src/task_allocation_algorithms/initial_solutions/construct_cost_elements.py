import numpy as np

from typing import Set, Tuple
from ...graph import Graph
from ...agent import AgentLoader

def manhattan_distance(loc1: Tuple[int, int], loc2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two locations."""
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

def construct_cost_elements(J: Set[Tuple], Rs: AgentLoader, G: Graph, method : str = "manhattan") -> tuple:
    """
    Compute and return the cost elements needed for allocation:
    - agent_start_cost_tensor
    - start_goal_dist
    - task_start_mask
    - task_goal_mask
    - start_locs
    - goal_locs
    - idx_to_task_id
    """
    allocated_tasks = set()
    for agent in Rs.agents:
        for task in agent.task_sequence:
            allocated_tasks.add(task[0])

    unallocated_tasks = [task for task in J if task[0] not in allocated_tasks]

    M = len(Rs.agents)  # Number of agents
    N = len(unallocated_tasks) # Number of tasks

    # Get all possible start and goal locations from unallocated tasks
    all_start_locs = set()
    all_goal_locs = set()
    for task in unallocated_tasks:
        all_start_locs.update(task[1])  # start_locations_frozenset
        all_goal_locs.update(task[2])  # goal_locations_frozenset

    # idx to task_id mapping
    idx_to_task_id = {idx: task[0] for idx, task in enumerate(unallocated_tasks)}
    
    # Convert to sorted lists for consistent indexing
    start_locs = sorted(list(all_start_locs))
    goal_locs = sorted(list(all_goal_locs))
    P = len(start_locs)
    Q = len(goal_locs)
    
    # Create location to index mappings
    start_loc_to_idx = {loc: idx for idx, loc in enumerate(start_locs)}
    goal_loc_to_idx = {loc: idx for idx, loc in enumerate(goal_locs)}

    # Get all locations currently allocated to tasks
    allocated_locs = set()
    for agent in Rs.agents:
        for task in agent.task_sequence:
            allocated_locs.update(task[1])
            allocated_locs.update(task[2])

    # 1. Build (P, Q) distance matrix between all start and goal locations
    if method == "manhattan":
        start_goal_dist = np.array([[manhattan_distance(s, g) for g in goal_locs] for s in start_locs])
    elif method == "shortest_path":
        start_goal_dist = np.array([[G.get_distance(s, g) for g in goal_locs] for s in start_locs])
    else:
        raise ValueError(f"Invalid cost calculation method: {method}")

    # 2. Build (N, P) task-start membership matrix (1 if task n has start_loc p and p not allocated, else 0)
    task_start_mask = np.zeros((N, P), dtype=np.float32)
    for n, task in enumerate(unallocated_tasks):
        for s in task[1]:
            if s not in allocated_locs:
                i = start_loc_to_idx[s]
                task_start_mask[n, i] = 1.0

    # 3. Build (N, Q) task-goal membership matrix (1 if task n has goal_loc q and q not allocated, else 0)
    task_goal_mask = np.zeros((N, Q), dtype=np.float32)
    for n, task in enumerate(unallocated_tasks):
        for g in task[2]:
            if g not in allocated_locs:
                j = goal_loc_to_idx[g]
                task_goal_mask[n, j] = 1.0

    # 4. Build (M, P) agent-start cost matrix
    agent_start_cost_tensor = np.full((M, P), np.inf)
    for m in range(M):
        # Determine agent's current position
        if len(Rs.agents[m].task_sequence) == 0:
            agent_pos = Rs.agents[m].state
        else:
            agent_pos = Rs.agents[m].task_sequence[-1][2]  # goal location of most recent task
        
        for i, s in enumerate(start_locs):
            if method == "manhattan":
                cost = manhattan_distance(agent_pos, s)
            elif method == "shortest_path":
                cost = G.get_distance(agent_pos, s)
            else:
                raise ValueError(f"Invalid cost calculation method: {method}")
            agent_start_cost_tensor[m, i] = cost

    return agent_start_cost_tensor, start_goal_dist, task_start_mask, task_goal_mask, start_locs, goal_locs, idx_to_task_id
