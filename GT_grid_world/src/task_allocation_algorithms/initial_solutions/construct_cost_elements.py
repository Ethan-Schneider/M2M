import numpy as np

from typing import Set, Tuple, List, Dict
from ...graph import Graph
from ...agent import AgentLoader

# Task type codes mirror those in case_request_generator.py / simulate.py:
#   0 = outbound (warehouse -> driveway)
#   1 = inbound  (driveway  -> warehouse)
#   2 = shuffle  (warehouse -> warehouse)
TASK_TYPE_OUTBOUND = 0
TASK_TYPE_INBOUND = 1
TASK_TYPE_SHUFFLE = 2

# Task types whose pickup is a warehouse SKU instance (so the outbound-style
# SKU-distribution cost over start_locs applies). The proper rearrangement
# objective for type=2 is roadmap section 1.6; for the 1.4 skeleton we let
# shuffle inherit the outbound cost so allocation produces finite values
# instead of np.inf.
WAREHOUSE_PICKUP_TASK_TYPES = frozenset({TASK_TYPE_OUTBOUND, TASK_TYPE_SHUFFLE})

# Task types whose dropoff is a warehouse-empty cell (so the inbound-style
# SKU-distribution cost over goal_locs applies). Same skeleton-vs-1.6 caveat.
WAREHOUSE_DROPOFF_TASK_TYPES = frozenset({TASK_TYPE_INBOUND, TASK_TYPE_SHUFFLE})


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
    
    if time_until_deadline > 60:
        # Deadline is far in the future, no urgency cost
        return 0
    elif time_until_deadline > 0:
        # Deadline is approaching, linear cost
        return int(60 - time_until_deadline)  # Linear increase as deadline approaches
    else:
        # Deadline has passed, quadratic cost
        overdue_time = abs(time_until_deadline)
        return int(60 + overdue_time)  # Quadratic penalty for overdue tasks

def construct_cost_elements(J: Dict[int, Tuple], Rs: AgentLoader, G: Graph, current_time: int, method : str = "manhattan") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]], Dict[int, int]]:
    """
    Compute the cost elements needed for allocation.
    Args:
        - J: Dict[task_id, (start_loc, goal_loc, deadline, sku_id, inbound)]
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
    allocated_task_ids = set()
    for agent in Rs.agents:
        for task in agent.task_sequence:
            allocated_task_ids.add(task[0])

    unallocated_task_ids = [task_id for task_id in J.keys() if task_id not in allocated_task_ids]

    M = len(Rs.agents)  # Number of agents
    N = len(unallocated_task_ids) # Number of tasks

    # Get all possible start and goal locations from unallocated tasks
    all_start_locs = set()
    all_goal_locs = set()
    for task_id in unallocated_task_ids:
        all_start_locs.update(J[task_id][0])  # start_locations_frozenset
        all_goal_locs.update(J[task_id][1])  # goal_locations_frozenset

    # idx to task_id mapping
    idx_to_task_id = {idx: task_id for idx, task_id in enumerate(unallocated_task_ids)}
    task_id_to_idx = {task_id: idx for idx, task_id in enumerate(unallocated_task_ids)}
    
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
        start_goal_dist = np.array([[manhattan_distance(s, g) for g in goal_locs] for s in start_locs])
    elif method == "shortest_path":
        start_goal_dist = np.array([[G.get_distance(s, g) for g in goal_locs] for s in start_locs])
    else:
        raise ValueError(f"Invalid cost calculation method: {method}")
    
    # print(f"Start Goal Distance Matrix: {start_goal_dist} with shape {start_goal_dist.shape}")
    # exit()

    # 2. Build (N, P) task-start membership matrix (1 if task n has start_loc p and p not unusable, else 0)
    task_start_mask = np.zeros((N, P), dtype=np.float32)
    for n, task_id in enumerate(unallocated_task_ids):
        for s in J[task_id][0]:
            if s not in allocated_locs:
                i = start_loc_to_idx[s]
                task_start_mask[n, i] = 1.0

    # 3. Build (N, Q) task-goal membership matrix (1 if task n has goal_loc q and q not unusable, else 0)
    task_goal_mask = np.zeros((N, Q), dtype=np.float32)
    for n, task_id in enumerate(unallocated_task_ids):
        for g in J[task_id][1]:
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
                agent_start_cost_tensor[m, i] = manhattan_distance(agent_pos, s)
            elif method == "shortest_path":
                agent_start_cost_tensor[m, i] = G.get_distance(agent_pos, s)
            else:
                raise ValueError(f"Invalid cost calculation method: {method}")
            
            # Find tasks that can use this start location and calculate deadline urgency
            # deadline_urgency_cost = 0
            # for n, task in enumerate(unallocated_tasks):
            #     if s in task[1] and task_start_mask[n, i] == 1.0:
            #         # This task can use this start location, add its deadline urgency
            #         task_deadline = task[3]
            #         urgency_cost = calculate_deadline_cost(task_deadline, current_time)
            #         deadline_urgency_cost = max(deadline_urgency_cost, urgency_cost)
            
            # Combine base cost with deadline urgency (positive urgency cost increases the negative base cost)
            # agent_start_cost_tensor[m, i] = 0.3*base_cost + 0.7*deadline_urgency_cost
            
    # print(f"Agent Start Cost Matrix: {agent_start_cost_tensor} with shape {agent_start_cost_tensor.shape}")
    
    # 5. Build (N) vector of task deadline costs
    task_deadline_costs = np.zeros(N)
    for n, task_id in enumerate(unallocated_task_ids):
        task_deadline_costs[n] = calculate_deadline_cost(J[task_id][2], current_time)

    # 6. Build (N, Q) inbound-style SKU distribution cost matrix.
    # Populated for tasks that drop off into the warehouse (inbound and
    # shuffle). Defaults to np.inf elsewhere so the allocator skips invalid
    # combos. Skeleton note for 1.4: shuffle reuses the inbound-style cost
    # here as a placeholder; the dedicated rearrangement objective lands in
    # roadmap section 1.6.
    inbound_sku_distribution_costs = np.full((N, Q), np.inf)
    for n, task_id in enumerate(unallocated_task_ids):
        task_type = J[task_id][4]
        if task_type in WAREHOUSE_DROPOFF_TASK_TYPES:
            for q in J[task_id][1]:
                if q not in unusable_locs:
                    j = goal_loc_to_idx[q]
                    inbound_sku_distribution_costs[n, j] = -1*G.query_sku_KD_trees(J[task_id][3], goal_locs[j], 1)[0]

    # 7. Build (N, P) outbound-style SKU distribution cost matrix.
    # Populated for tasks that pick up from the warehouse (outbound and
    # shuffle). Same skeleton-vs-1.6 caveat as above. fast_greedy currently
    # routes any task with type != 1 through this matrix, so populating
    # type=2 here is what keeps shuffle-task allocation cost finite.
    outbound_sku_distribution_costs = np.full((N, P), np.inf)
    for n, task_id in enumerate(unallocated_task_ids):
        task_type = J[task_id][4]
        if task_type in WAREHOUSE_PICKUP_TASK_TYPES:
            for p in J[task_id][0]:
                if p not in allocated_locs:
                    i = start_loc_to_idx[p]
                    # Second-closest because the closest is the start cell itself.
                    outbound_sku_distribution_costs[n, i] = G.query_sku_KD_trees(J[task_id][3], start_locs[i], 2)[0][1]


    # 8. Build vector of size (M) which includes the estimated time for the agent to complete the task sequence
    agent_task_sequence_time = np.zeros(M)
    for m in range(M):
        if len(Rs.agents[m].task_sequence) == 0:
            continue
        agent_task_sequence_time[m] = G.get_distance(Rs.agents[m].state, Rs.agents[m].task_sequence[0][1])
        for i in range(1, len(Rs.agents[m].task_sequence)):
            #add the distance between the goal of the previous task and the start of the current task
            agent_task_sequence_time[m] += G.get_distance(Rs.agents[m].task_sequence[i-1][2], Rs.agents[m].task_sequence[i][1])
            #add the distance between the start of the current task to the goal of the current task
            agent_task_sequence_time[m] += G.get_distance(Rs.agents[m].task_sequence[i][1], Rs.agents[m].task_sequence[i][2])

    return agent_start_cost_tensor, start_goal_dist, task_start_mask, task_goal_mask, start_locs, goal_locs, idx_to_task_id, task_id_to_idx, task_deadline_costs, inbound_sku_distribution_costs, outbound_sku_distribution_costs, agent_task_sequence_time
