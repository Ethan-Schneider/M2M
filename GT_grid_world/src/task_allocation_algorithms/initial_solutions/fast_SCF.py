import numpy as np
import time
from typing import Set, Tuple, List, Dict
from ...graph import Graph
from ...agent import AgentLoader
from ...analysis.statistics import Stats
from .construct_cost_elements import construct_cost_elements, manhattan_distance

def fast_SCF_allocation(S : Stats, G : Graph, Rs : AgentLoader, start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]], idx_to_task_id: Dict[int, int], J, method : str = "manhattan",
                         agent_start_cost_tensor=None, start_goal_dist=None, task_start_mask=None, task_goal_mask=None, cost_lookup=None) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], float]:
    """
    Perform second coordinate fixing (SCF) greedy allocation of tasks to agents based on minimum cost elements, without constructing the full (M, N, P, Q) tensor.
    This is a batched greedy algorithm that allocates one task per agent per batch, repeating until all tasks are allocated.
    Ensures that no location is used as a start or goal more than once in the entire allocation process.
    
    Args:
        S: Statistics object
        G: Graph object
        Rs: AgentLoader containing all agents
        start_locs: List of start locations
        goal_locs: List of goal locations
        idx_to_task_id: Dict mapping task indices to task IDs
        method: Cost calculation method
        agent_start_cost_tensor: (M, P) array
        start_goal_dist: (P, Q) array
        task_start_mask: (N, P) array
        task_goal_mask: (N, Q) array
        J: Set of tasks
    Returns:
        Tuple containing:
        - AgentLoader object with updated task sequences
        - List of tuples (m, n, p, q) representing allocations
        - Total cost of all allocations
    """
    M = len(Rs.agents)
    N = len(idx_to_task_id)
    P = len(start_locs)

    allocations = []
    total_cost = 0.0
    total_argmin_time = 0.0
    total_update_time = 0.0

    # Copy masks so we can update them
    task_start_mask_ = task_start_mask.copy()
    task_goal_mask_ = task_goal_mask.copy()
    assigned_tasks = set()

    # Iterate over all tasks
    for n in range(N):
        argmin_tik = time.time()
        min_cost = -np.inf

        valid_p = np.where(task_start_mask_[n] == 1)[0]
        valid_q = np.where(task_goal_mask_[n] == 1)[0]
        if len(valid_p) == 0 or len(valid_q) == 0:
            continue

        # Find the task with the minimum cost for this agent
        best = None
        for m in range(M):
            # Vectorized cost computation for all valid (p, q) pairs
            total_costs = agent_start_cost_tensor[m, valid_p][:, None] + start_goal_dist[np.ix_(valid_p, valid_q)]               

            min_idx = np.argmax(total_costs)
            min_cost_m = total_costs.flat[min_idx]
            if min_cost_m > min_cost:
                min_cost = min_cost_m
                p_idx, q_idx = np.unravel_index(min_idx, total_costs.shape)
                best = (m, valid_p[p_idx], valid_q[q_idx])
        total_argmin_time += time.time() - argmin_tik
        if best is None or min_cost == -np.inf:
            break

        # Add the task to the allocation
        m, p, q = best
        allocations.append((int(m), idx_to_task_id[int(n)], int(p), int(q)))

        if cost_lookup is not None:
            # Store the cost in the lookup table
            cost_lookup[(int(m), idx_to_task_id[int(n)], int(p), int(q))] = int(min_cost)

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

        # Update agent's start cost tensor
        # 4. Update costs for agent-start allocation for agent m
        for p in range(P):
            if method == "manhattan":
                cost = -1.0 * manhattan_distance(goal_locs[q], start_locs[p])
            elif method == "shortest_path":
                cost = -1.0 * G.get_distance(goal_locs[q], start_locs[p])
            else:
                raise ValueError(f"Invalid cost calculation method: {method}")
            agent_start_cost_tensor[m, p] = cost

        total_update_time += time.time() - update_tik

    print(f"Total argmin time: {total_argmin_time}")
    print(f"Total update time: {total_update_time}")
    return Rs, allocations, total_cost


def fast_SCF_call(S: Stats, G: Graph, Rs: AgentLoader, J: Set[Tuple], current_time: int, method : str = "manhattan") -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], float]:
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

    allocation_tik = time.time()
    # Perform greedy allocation
    Rs, allocations, cost = fast_SCF_allocation(
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

    print(f"Number of allocations: {len(allocations)}")
    print(f"Total cost: {cost}")

    return Rs, allocations, cost
