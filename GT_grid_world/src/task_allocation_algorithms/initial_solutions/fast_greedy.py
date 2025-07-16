import numpy as np
import time
from typing import Set, Tuple, List, Dict
from ...graph import Graph
from ...agent import AgentLoader
from ...analysis.statistics import Stats
from .construct_cost_elements import construct_cost_elements, manhattan_distance

def fast_greedy_allocation(S : Stats, G : Graph, Rs : AgentLoader, start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]], idx_to_task_id: Dict[int, int], J, method : str = "manhattan",
                         agent_start_cost_tensor=None, start_goal_dist=None, task_start_mask=None, task_goal_mask=None, cost_lookup=None) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], float]:
    """
    Perform first coordinate fixing (FCF) greedy allocation of tasks to agents based on minimum cost elements, without constructing the full (M, N, P, Q) tensor.
    This is a batched greedy algorithm that allocates one task per agent per batch, repeating until all tasks are allocated.
    Ensures that no location is used as a start or goal more than once in the entire allocation process.
    
    Args:
        S: Statistics object
        G: Graph object
        Rs: AgentLoader containing all agents
        start_locs: List of start locations
        goal_locs: List of goal locations
        idx_to_task_id: Dict mapping task indices to task IDs
        J: Set of tasks
        method: Cost calculation method
        agent_start_cost_tensor: (M, P) array
        start_goal_dist: (P, Q) array
        task_start_mask: (N, P) array
        task_goal_mask: (N, Q) array
    Returns:
        Tuple containing:
        - AgentLoader object with updated task sequences
        - List of tuples (m, n, p, q) representing allocations
        - Total cost of all allocations
    """
    M = len(Rs.agents)
    N = len(idx_to_task_id)
    P = len(start_locs)
    Q = len(goal_locs)
    allocations = []
    total_cost = 0.0
    total_argmin_time = 0.0
    total_update_time = 0.0

    # Copy masks so we can update them
    task_start_mask_ = task_start_mask.copy()
    task_goal_mask_ = task_goal_mask.copy()
    assigned_tasks = set()

    iteration = 0

    while True:
        iteration += 1
        # If all tasks are assigned, if no start or goal locations are left, break
        if len(assigned_tasks) == N:
            print(f"All tasks assigned: {len(assigned_tasks)}")
            break

        # If all start or goal locations are used, break
        if np.all(task_start_mask_ == 0) or np.all(task_goal_mask_ == 0):
            print(f"No start or goal locations left")
            break

        # Iterate over each task and find the best allocation
        argmin_tik = time.time()
        min_cost = -1 * np.inf
        best = None

        # Iterate over each task
        for n in range(N):

            # If the task is already assigned, skip
            if n in assigned_tasks:
                continue

            # Find all valid start and goal locations for the task
            valid_p = np.where(task_start_mask_[n] == 1)[0]
            valid_q = np.where(task_goal_mask_[n] == 1)[0]

            # If there are no valid start or goal locations, skip
            if len(valid_p) == 0 or len(valid_q) == 0:
                continue

            # Mask out all invalid start and goal locations
            agent_costs = agent_start_cost_tensor[:, valid_p]
            sg_costs = start_goal_dist[np.ix_(valid_p, valid_q)]

            # Compute the total cost for all valid (p, q) pairs for the task
            total_costs = agent_costs[:, :, None] + sg_costs[None, :, :]

            # Find argmax of total_costs, if there are multiple max values, choose one randomly
            min_idx = np.argmax(total_costs)
            min_cost_n = total_costs.flat[min_idx]

            # If new best cost is found, update the best allocation
            if min_cost_n > min_cost:
                # If there are multiple min costs, choose one randomly
                if np.sum(total_costs == min_cost_n) > 1:
                    # Restructure max location into list of tuples
                    min_locations = np.where(total_costs == min_cost_n)
                    min_locations_list = []
                    for i in range(len(min_locations[0])):
                        min_locations_list.append((int(min_locations[0][i]), int(min_locations[1][i]), int(min_locations[2][i])))

                    min_idx = np.random.choice(range(len(min_locations_list)), 1)[0]
                    m_idx, p_idx, q_idx = min_locations_list[min_idx]

                    if min_cost_n > min_cost:
                        min_cost = min_cost_n
                        m_idx, p_idx, q_idx = m_idx, p_idx, q_idx
                        best = (m_idx, n, valid_p[p_idx], valid_q[q_idx])
                # If there is only one min cost, update the best allocation
                else:
                    min_cost = min_cost_n
                    m_idx, p_idx, q_idx = np.unravel_index(min_idx, total_costs.shape)
                    best = (m_idx, n, valid_p[p_idx], valid_q[q_idx])

        # If no best task is found, break
        total_argmin_time += time.time() - argmin_tik
        if best is None or min_cost == -1 * np.inf:
            print(f"No best task found")
            break

        # Add the best task to the allocation
        m, n, p, q = best
        allocations.append((int(m), idx_to_task_id[int(n)], int(p), int(q)))

        # If cost lookup is provided, store the cost in the lookup table
        if cost_lookup is not None:
            # Store the cost in the lookup table
            cost_lookup[(int(m), idx_to_task_id[int(n)], int(p), int(q))] = int(min_cost)
        
        # Update the total cost
        total_cost += min_cost
        assigned_tasks.add(n)

        # Update statistics
        S.append_early_task_ids(idx_to_task_id[int(n)])
        S.add_actual_distance(idx_to_task_id[int(n)])
        S.add_actual_pickup_distance(idx_to_task_id[int(n)])
        S.add_actual_duration(idx_to_task_id[int(n)])
        S.add_actual_pickup_duration(idx_to_task_id[int(n)])

        # Update agent's task sequence
        deadline = next(task[3] for task in J if task[0] == idx_to_task_id[int(n)])
        Rs.agents[m].task_sequence.append((idx_to_task_id[int(n)], start_locs[p], goal_locs[q], deadline))
        if Rs.agents[m].status == 0:
            Rs.agents[m].status = 1
        update_tik = time.time()

        # Invalidate this task, start, and goal for all future agents
        task_start_mask_[:, p] = 0
        task_goal_mask_[:, q] = 0
        task_start_mask_[n, :] = 0
        task_goal_mask_[n, :] = 0
        total_update_time += time.time() - update_tik

        # 4. Update costs for agent-start allocation for agent m
        for p_ in range(P):
            if method == "manhattan":
                cost = -1.0 * manhattan_distance(goal_locs[q], start_locs[p_])
            elif method == "shortest_path":
                cost = -1.0 * G.get_distance(goal_locs[q], start_locs[p_])
            else:
                raise ValueError(f"Invalid cost calculation method: {method}")
            agent_start_cost_tensor[m, p_] = cost

        total_update_time += time.time() - update_tik

    print(f"Total argmin time: {total_argmin_time}")
    print(f"Total update time: {total_update_time}")
    return Rs, allocations, total_cost


def fast_greedy_call(S: Stats, G: Graph, Rs: AgentLoader, J: Set[Tuple], current_time: int, method : str = "manhattan") -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], float]:
    """
    Multi-Agent to Multi-Task Large Neighborhood Search algorithm (batched greedy version).
    Allocates one task per agent per batch, repeating until all tasks are allocated.
    Ensures that no location is used as a start or goal more than once in the entire allocation process.
    
    Args:
        S: Statistics object
        G: Graph object
        Rs: AgentLoader containing all agents
        J: Set of tasks
        current_time: Current timestep for deadline calculations
        method: Cost calculation method
    """
    tik = time.time()
    num_allocated_tasks = 0
    for agent in Rs.agents:
        num_allocated_tasks += len(agent.task_sequence)
    if len(J) == num_allocated_tasks:  # No tasks to assign
        print(f"No tasks to assign")
        return Rs, [], 0.0
    
    # Clear all but the first task in each agent's task sequence
    for agent in Rs.agents:
        while len(agent.task_sequence) > 1:
            agent.task_sequence.pop(-1)

    total_construct_time = 0.0
    total_allocation_time = 0.0

    construct_tik = time.time()
    agent_start_cost_tensor, start_goal_dist, task_start_mask, task_goal_mask, start_locs, goal_locs, idx_to_task_id = construct_cost_elements(J, Rs, G, current_time, method)
    total_construct_time += time.time() - construct_tik

    # print(f"Min cost agent-start cost: {np.min(agent_start_cost_tensor)}")
    # print(f"Min cost start-goal cost: {np.min(start_goal_dist)}")
    # print(f"Number of available start locations: {np.sum(task_start_mask)}")
    # print(f"Number of available goal locations: {np.sum(task_goal_mask)}")
    # print(f"Number of start locations: {len(start_locs)}")
    # print(f"Number of goal locations: {len(goal_locs)}")
    # print(f"Number of tasks: {len(J)}")
    # for task in J:
    #     print(f"Task {task[0]} has {len(task[1])} start locations and {len(task[2])} goal locations")

    allocation_tik = time.time()
    # Perform greedy allocation
    Rs, allocations, cost = fast_greedy_allocation(
        S, G, Rs, start_locs, goal_locs, idx_to_task_id, J, method,
        agent_start_cost_tensor=agent_start_cost_tensor,
        start_goal_dist=start_goal_dist,
        task_start_mask=task_start_mask,
        task_goal_mask=task_goal_mask
    )
    total_allocation_time += time.time() - allocation_tik
        
    tok = time.time()
    print(f"Fast greedy allocation (batched) total time {tok-tik}s")
    print(f"Total construct time: {total_construct_time}")
    print(f"Total allocation time: {total_allocation_time}")
    print(f"Agent task sequences:")
    for agent in Rs.agents:
        print(f"Agent {agent.id}: {agent.task_sequence}")
    return Rs, allocations, cost
