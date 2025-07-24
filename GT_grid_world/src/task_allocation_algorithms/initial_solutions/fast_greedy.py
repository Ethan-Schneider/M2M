import numpy as np
import time
from typing import Tuple, List, Dict
from ...graph import Graph
from ...agent import AgentLoader
from ...analysis.statistics import Stats
from .construct_cost_elements import construct_cost_elements, manhattan_distance

def fast_greedy_allocation(S : Stats, G : Graph, Rs : AgentLoader, start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]], idx_to_task_id: Dict[int, int], J, method : str = "manhattan",
                         agent_start_cost_tensor=None, start_goal_dist=None, task_start_mask=None, task_goal_mask=None, cost_lookup=None, task_deadline_costs=None, inbound_sku_distribution_costs=None, 
                         outbound_sku_distribution_costs=None, base_cost_weight=1.0, deadline_weight=0.0, sku_distribution_weight=0.0, agent_task_sequence_time=None, current_time=0) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], float, Dict[Tuple[int, int, int, int], int]]:
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
        task_deadline_costs: (N) array
        inbound_sku_distribution_costs: (N, Q) array
        outbound_sku_distribution_costs: (N, P) array
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
        best_cost = -1 * np.inf
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

            # Calculate base costs
            deadline = J[idx_to_task_id[int(n)]][2]
            if deadline_weight > 0.0:
                # If deadline has not passed
                if deadline - current_time > 0:
                    base_costs = -1*deadline_weight*(deadline - current_time) + base_cost_weight*((agent_costs[:, :, None] + sg_costs[None, :, :]) + agent_task_sequence_time[:, None, None])
                    # base_costs = base_cost_weight*(agent_costs[:, :, None] + sg_costs[None, :, :]) - deadline_weight*(deadline - (current_time - ((agent_costs[:, :, None] + sg_costs[None, :, :]) + agent_task_sequence_time[:, None, None])))
                # If deadline has passed
                else:
                    base_costs = deadline_weight*np.abs(deadline - current_time) + base_cost_weight*((agent_costs[:, :, None] + sg_costs[None, :, :]) + agent_task_sequence_time[:, None, None])
            else:
                base_costs = base_cost_weight*(agent_costs[:, :, None] + sg_costs[None, :, :])

            # If inbound task, add inbound sku distribution costs
            if J[idx_to_task_id[int(n)]][4] == 1:
                inbound_sku_distribution_costs_n = inbound_sku_distribution_costs[n, valid_q]
                total_costs = base_costs + sku_distribution_weight*inbound_sku_distribution_costs_n[None, None, :]
            # If outbound task, add outbound sku distribution costs
            else:
                outbound_sku_distribution_costs_n = outbound_sku_distribution_costs[n, valid_p]
                total_costs = base_costs + sku_distribution_weight*outbound_sku_distribution_costs_n[None, :, None]

            # Find argmax of total_costs, if there are multiple max values, choose one randomly
            new_idx = np.argmax(total_costs)
            new_cost = total_costs.flat[new_idx]

            # If new best cost is found, update the best allocation
            if new_cost > best_cost:
                # If there are multiple max costs, choose one randomly
                if np.sum(total_costs == new_cost) > 1:
                    # Restructure max location into list of tuples
                    max_locations = np.where(total_costs == new_cost)
                    max_locations_list = [(int(max_locations[0][i]), int(max_locations[1][i]), int(max_locations[2][i])) for i in range(len(max_locations[0]))]
                    
                    random_idx = np.random.choice(range(len(max_locations_list)), 1)[0]
                    m_idx, p_idx, q_idx = max_locations_list[random_idx]
                    best_cost = new_cost
                    best = (m_idx, n, valid_p[p_idx], valid_q[q_idx])
                # If there is only one max cost, update the best allocation
                else:
                    best_cost = new_cost
                    m_idx, p_idx, q_idx = np.unravel_index(new_idx, total_costs.shape)
                    best = (m_idx, n, valid_p[p_idx], valid_q[q_idx])

        # If no best task is found, break
        total_argmin_time += time.time() - argmin_tik
        if best is None or best_cost == -1 * np.inf:
            print(f"No best task found")
            break

        # Add the best task to the allocation
        m, n, p, q = best
        allocations.append((int(m), idx_to_task_id[int(n)], int(p), int(q)))

        # Store the cost in the lookup table
        if cost_lookup is not None:
            cost_lookup[(int(m), idx_to_task_id[int(n)], int(p), int(q))] = int(best_cost)
        
        # Update the total cost
        total_cost += best_cost
        assigned_tasks.add(n)

        # Update statistics
        S.append_early_task_ids(idx_to_task_id[int(n)])
        S.add_actual_distance(idx_to_task_id[int(n)])
        S.add_actual_pickup_distance(idx_to_task_id[int(n)])
        S.add_actual_duration(idx_to_task_id[int(n)])
        S.add_actual_pickup_duration(idx_to_task_id[int(n)])

        # Update agent's task sequence
        deadline = J[idx_to_task_id[int(n)]][2]
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

        # Update agent task sequence time
        if len(Rs.agents[m].task_sequence) > 1:
            agent_task_sequence_time[m] -= G.get_distance(Rs.agents[m].task_sequence[-2][2], Rs.agents[m].task_sequence[-1][1])
            agent_task_sequence_time[m] -= G.get_distance(Rs.agents[m].task_sequence[-1][1], Rs.agents[m].task_sequence[-1][2])
        else:
            agent_task_sequence_time[m] -= G.get_distance(Rs.agents[m].state, Rs.agents[m].task_sequence[0][1])
            agent_task_sequence_time[m] -= G.get_distance(Rs.agents[m].task_sequence[0][1], Rs.agents[m].task_sequence[0][2])

        total_update_time += time.time() - update_tik

    print(f"Total argmin time: {total_argmin_time}")
    print(f"Total update time: {total_update_time}")
    return Rs, allocations, total_cost


def fast_greedy_call(S: Stats, G: Graph, Rs: AgentLoader, J: Dict[int, Tuple], 
                     current_time: int, method : str = "manhattan", base_cost_weight=1.0, deadline_weight=0.0, sku_distribution_weight=0.0) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], float]:
    """
    Multi-Agent to Multi-Task Large Neighborhood Search algorithm (batched greedy version).
    Allocates one task per agent per batch, repeating until all tasks are allocated.
    Ensures that no location is used as a start or goal more than once in the entire allocation process.
    
    Args:
        S: Statistics object
        G: Graph object
        Rs: AgentLoader containing all agents
        J: Dict of tasks
        current_time: Current timestep for deadline calculations
        method: Cost calculation method
        base_cost_weight: Weight for base cost
        deadline_weight: Weight for deadline
        sku_distribution_weight: Weight for sku distribution
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
    agent_start_cost_tensor, start_goal_dist, task_start_mask, task_goal_mask, start_locs, goal_locs, idx_to_task_id, task_id_to_idx, task_deadline_costs, inbound_sku_distribution_costs, outbound_sku_distribution_costs, agent_task_sequence_time = construct_cost_elements(J, Rs, G, current_time, method)
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
        task_goal_mask=task_goal_mask,
        task_deadline_costs=task_deadline_costs,
        inbound_sku_distribution_costs=inbound_sku_distribution_costs,
        outbound_sku_distribution_costs=outbound_sku_distribution_costs,
        base_cost_weight=base_cost_weight,
        deadline_weight=deadline_weight,
        sku_distribution_weight=sku_distribution_weight,
        agent_task_sequence_time=agent_task_sequence_time,
        current_time=current_time
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
