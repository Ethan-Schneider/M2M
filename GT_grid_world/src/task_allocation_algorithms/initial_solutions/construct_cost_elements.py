import numpy as np

from typing import Set, Tuple, List, Dict
from ...graph import Graph
from ...agent import AgentLoader

def manhattan_distance(loc1: Tuple[int, int], loc2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two locations."""
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

def calculate_deadline_cost(deadline: int, current_time: int) -> int:
    """
    Calculate deadline-based urgency cost using piecewise function.
    
    Args:
        deadline: Task deadline (integer timestep)
        current_time: Current timestep
    
    Returns:
        Positive integer cost based on deadline urgency
    """
    time_until_deadline = deadline - current_time
    
    if time_until_deadline > 30:
        # Deadline is far in the future, no urgency cost
        return 0
    elif time_until_deadline > 0:
        # Deadline is approaching within 10 seconds, linear cost
        return int(30 - time_until_deadline)  # Linear increase as deadline approaches
    else:
        # Deadline has passed, quadratic cost
        overdue_time = abs(time_until_deadline)
        return int(30 + overdue_time**2)  # Quadratic penalty for overdue tasks

def construct_cost_elements(J: Set[Tuple], Rs: AgentLoader, G: Graph, current_time: int, method : str = "manhattan") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]], Dict[int, int]]:
    """
    Compute the cost elements needed for allocation.
    Args:
        - J: Set[Tuple]
        - Rs: AgentLoader
        - G: Graph
        - current_time: Current timestep for deadline calculations
        - method: str
    Returns:
        - agent_start_cost_tensor: (M, P)
        - start_goal_dist: (P, Q)
        - task_start_mask: (N, P)
        - task_goal_mask: (N, Q)
        - start_locs: List[Tuple[int, int]]
        - goal_locs: List[Tuple[int, int]]
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

    # print(f"Number of start locations: {P}")
    # print(f"Number of goal locations: {Q}")
    # print(f"Number of tasks: {N}")
    # print(f"Number of agents: {M}")
    
    # Create location to index mappings
    start_loc_to_idx = {loc: idx for idx, loc in enumerate(start_locs)}
    goal_loc_to_idx = {loc: idx for idx, loc in enumerate(goal_locs)}

    # Get all locations currently allocated to tasks
    allocated_locs = set()
    for agent in Rs.agents:
        for task in agent.task_sequence:
            allocated_locs.add(task[1])
            allocated_locs.add(task[2])

    # Get all locations occupied by items in warehouse and driveway
    warehouse_occupied_locs = set(G.warehouse.get_full_locations())
    driveway_occupied_locs = set(G.driveway.get_full_locations())

    # Combine all unusable locations
    unusable_locs = allocated_locs | warehouse_occupied_locs | driveway_occupied_locs

    # 1. Build (P, Q) distance matrix between all start and goal locations
    if method == "manhattan":
        start_goal_dist = np.array([[-1.0 * manhattan_distance(s, g) for g in goal_locs] for s in start_locs])
    elif method == "shortest_path":
        start_goal_dist = np.array([[-1.0 * G.get_distance(s, g) for g in goal_locs] for s in start_locs])
    else:
        raise ValueError(f"Invalid cost calculation method: {method}")

    # 2. Build (N, P) task-start membership matrix (1 if task n has start_loc p and p not unusable, else 0)
    task_start_mask = np.zeros((N, P), dtype=np.float32)
    for n, task in enumerate(unallocated_tasks):
        for s in task[1]:
            if s not in allocated_locs:
                i = start_loc_to_idx[s]
                task_start_mask[n, i] = 1.0

    # 3. Build (N, Q) task-goal membership matrix (1 if task n has goal_loc q and q not unusable, else 0)
    task_goal_mask = np.zeros((N, Q), dtype=np.float32)
    for n, task in enumerate(unallocated_tasks):
        for g in task[2]:
            if g not in unusable_locs:
                j = goal_loc_to_idx[g]
                task_goal_mask[n, j] = 1.0

    # 4. Build (M, P) agent-start cost matrix with deadline urgency costs
    agent_start_cost_tensor = np.full((M, P), np.inf)
    for m in range(M):
        # Determine agent's current position
        if len(Rs.agents[m].task_sequence) == 0:
            agent_pos = Rs.agents[m].state
        else:
            agent_pos = Rs.agents[m].task_sequence[-1][2]  # goal location of most recent task
        
        for i, s in enumerate(start_locs):
            # Base cost (negative for argmax logic)
            if method == "manhattan":
                base_cost = -1.0 * manhattan_distance(agent_pos, s)
            elif method == "shortest_path":
                base_cost = -1.0 * G.get_distance(agent_pos, s)
            else:
                raise ValueError(f"Invalid cost calculation method: {method}")
            
            # Find tasks that can use this start location and calculate deadline urgency
            deadline_urgency_cost = 0
            # for n, task in enumerate(unallocated_tasks):
            #     if s in task[1] and task_start_mask[n, i] == 1.0:
            #         # This task can use this start location, add its deadline urgency
            #         task_deadline = task[3]
            #         urgency_cost = calculate_deadline_cost(task_deadline, current_time)
            #         deadline_urgency_cost = max(deadline_urgency_cost, urgency_cost)
            
            # Combine base cost with deadline urgency (positive urgency cost increases the negative base cost)
            agent_start_cost_tensor[m, i] = base_cost + 0.1*deadline_urgency_cost
            
    # print(f"Agent start cost tensor: {agent_start_cost_tensor}")
    # print(f"Created tensors")
    return agent_start_cost_tensor, start_goal_dist, task_start_mask, task_goal_mask, start_locs, goal_locs, idx_to_task_id
