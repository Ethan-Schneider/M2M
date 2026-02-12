import numpy as np
from typing import List, Tuple, Dict, Set
from ...agent import AgentLoader
from ...analysis.statistics import Stats
from ...graph import Graph
from ...utils import manhattan_distance

import time

def greedy_repair(S: Stats, G: Graph, agent_start_cost_tensor: np.ndarray, start_goal_dist: np.ndarray, task_start_mask: np.ndarray, task_goal_mask: np.ndarray, Rs: AgentLoader,
                 start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]],
                 idx_to_task_id: Dict[int, int], temp_allocations: List[Tuple[int, int, int, int]],
                 method: str = "manhattan", J: Dict[int, Tuple] = None, cost_lookup: Dict[Tuple[int, int, int, int], int] = None,
                 inbound_sku_distribution_costs: np.ndarray = None, outbound_sku_distribution_costs: np.ndarray = None,
                 base_cost_weight: float = 1.0, deadline_weight: float = 0.0, sku_distribution_weight: float = 0.0,
                 agent_task_sequence_time: np.ndarray = None, current_time: int = 0, agent_task_sequence_limit: int = -1) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], float]:
    """
    Greedily repair a solution by iteratively assigning the minimum cost allocation using cost elements.
    Args:
        S: Statistics object
        G: Graph object
        agent_start_cost_tensor: (M, P) array
        start_goal_dist: (P, Q) array
        task_start_mask: (N, P) array
        task_goal_mask: (N, Q) array
        Rs: AgentLoader containing all agents
        start_locs: List of start locations
        goal_locs: List of goal locations
        idx_to_task_id: Mapping from tensor indices to task IDs
        temp_allocations: List of current allocations (agent_idx, task_idx, start_idx, goal_idx)
        method: Cost calculation method ("manhattan" or "shortest_path")
        current_time: Current timestep for deadline calculations
    Returns:
        Tuple containing:
        - AgentLoader with updated task sequences
        - List of allocations (m, n, p, q)
        - Total cost of all allocations
    """
    total_cost = 0.0
    M = len(Rs.agents)
    N = len(idx_to_task_id.keys())
    P = len(start_locs)

    total_update_time = 0.0
    total_find_best_task_time = 0.0

    while True:
        # If no start or goal locations are left, break
        if np.all(task_start_mask == 0) or np.all(task_goal_mask == 0):
            break
        
        # Check if we can assign any more tasks
        can_assign_any = False
        for n in range(N):
            if idx_to_task_id[int(n)] in [key[1] for key in cost_lookup.keys()]:
                continue
            valid_p = np.where(task_start_mask[n] == 1)[0]
            valid_q = np.where(task_goal_mask[n] == 1)[0]
            if len(valid_p) > 0 and len(valid_q) > 0:
                can_assign_any = True
                break
        
        if not can_assign_any:
            break

        best_cost = np.inf
        best = None

        tik = time.time()
        # print(f"idx to task id: {idx_to_task_id}")
        # print(f"N: {N}")
        # print(f"cost lookup: {cost_lookup}")
        # For each task, find the best (m, n, p, q)
        for n in range(N):
            # If the task is already assigned, skip, keys are (m, n, p, q)
            if idx_to_task_id[int(n)] in [key[1] for key in cost_lookup.keys()]:
                continue

            # Find all valid start and goal locations for the task
            valid_p = np.where(task_start_mask[n] == 1)[0]
            valid_q = np.where(task_goal_mask[n] == 1)[0]

            # If there are no valid start or goal locations, skip
            if len(valid_p) == 0 or len(valid_q) == 0:
                continue

            # Compute the total cost for all valid (p, q) pairs for the task
            agent_costs = agent_start_cost_tensor[:, valid_p]  # (M, len(valid_p))
            sg_costs = start_goal_dist[np.ix_(valid_p, valid_q)]  # (len(valid_p), len(valid_q))

            # Calculate base costs with deadline and agent_task_sequence_time considerations
            # deadline = J[idx_to_task_id[int(n)]][2]
            # if deadline_weight > 0.0:
            #     # If deadline has not passed
            #     if deadline - current_time > 0:
            #         base_costs = -1*deadline_weight*(deadline - current_time) + base_cost_weight*((agent_costs[:, :, None] + sg_costs[None, :, :]) + agent_task_sequence_time[:, None, None])
            #     # If deadline has passed
            #     else:
            #         base_costs = deadline_weight*np.abs(deadline - current_time) + base_cost_weight*((agent_costs[:, :, None] + sg_costs[None, :, :]) + agent_task_sequence_time[:, None, None])
            # else:
            base_costs = base_cost_weight*(agent_costs[:, :, None] + sg_costs[None, :, :])

            if sku_distribution_weight > 0:
                # Add sku distribution costs
                if J[idx_to_task_id[int(n)]][4] == 1:
                    inbound_sku_distribution_costs_n = inbound_sku_distribution_costs[n, valid_q]
                    total_costs = base_costs + sku_distribution_weight*inbound_sku_distribution_costs_n[None, None, :]
                else:
                    outbound_sku_distribution_costs_n = outbound_sku_distribution_costs[n, valid_p]
                    total_costs = base_costs + sku_distribution_weight*outbound_sku_distribution_costs_n[None, :, None]
            else:
                total_costs = base_costs
            
            # Iterate over each agent, if task sequence limit is reached, set total_costs[m, :, :] to -inf
            if agent_task_sequence_limit > -1:
                for m in range(M):
                    if len(Rs.agents[m].task_sequence) >= agent_task_sequence_limit:
                        total_costs[m, :, :] = np.inf
            # If all total_costs are -inf, break
            if np.all(total_costs == np.inf):
                break

            # Find the index of the maximum cost (since costs are negative, this minimizes distance)
            new_idx = np.argmin(total_costs)
            new_cost = total_costs.flat[new_idx]

            # If the new cost is greater than the current best cost, update the best allocation
            if new_cost < best_cost:
                # If there are multiple maximum costs, choose one randomly
                if np.sum(total_costs == new_cost) > 1:
                    max_locations = np.where(total_costs == new_cost)
                    max_locations_list = [(int(max_locations[0][i]), int(max_locations[1][i]), int(max_locations[2][i])) for i in range(len(max_locations[0]))]

                    random_idx = np.random.choice(range(len(max_locations_list)), 1)[0]
                    m_idx, p_idx, q_idx = max_locations_list[random_idx]

                    best_cost = new_cost
                    best = (m_idx, n, valid_p[p_idx], valid_q[q_idx])
                else:
                    best_cost = new_cost
                    m_idx, p_idx, q_idx = np.unravel_index(new_idx, total_costs.shape)
                    best = (m_idx, n, valid_p[p_idx], valid_q[q_idx])

        total_find_best_task_time += time.time() - tik

        if best is None or best_cost == np.inf:
            break

        tik = time.time()

        # Add the best task to the allocation
        m, n, p, q = best
        cost_lookup[(int(m), idx_to_task_id[int(n)], int(p), int(q))] = int(best_cost)
        total_cost += best_cost

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

        # Invalidate this task, start, and goal for all future agents
        task_start_mask[:, p] = 0
        task_goal_mask[:, q] = 0
        task_start_mask[n, :] = 0
        task_goal_mask[n, :] = 0

        # Update costs for agent-start allocation for agent m
        for p_ in range(P):
            if method == "manhattan":
                cost = manhattan_distance(goal_locs[q], start_locs[p_])
            elif method == "shortest_path":
                cost = G.get_distance(goal_locs[q], start_locs[p_])
            else:
                raise ValueError(f"Invalid cost calculation method: {method}")
            agent_start_cost_tensor[m, p_] = cost
            
        # If agent task sequence limit is reached, set agent_start_cost_temsor[m, :] to inf
        # if agent_task_sequence_limit > 0 and len(Rs.agents[m].task_sequence) >= agent_task_sequence_limit:
        #     agent_start_cost_tensor[m, :] = -1 * np.inf

        # Update agent task sequence time
        if len(Rs.agents[m].task_sequence) > 1:
            agent_task_sequence_time[m] += G.get_distance(Rs.agents[m].task_sequence[-2][2], Rs.agents[m].task_sequence[-1][1])
            agent_task_sequence_time[m] += G.get_distance(Rs.agents[m].task_sequence[-1][1], Rs.agents[m].task_sequence[-1][2])
        else:
            agent_task_sequence_time[m] += G.get_distance(Rs.agents[m].state, Rs.agents[m].task_sequence[0][1])
            agent_task_sequence_time[m] += G.get_distance(Rs.agents[m].task_sequence[0][1], Rs.agents[m].task_sequence[0][2])

        total_update_time += time.time() - tik

    return Rs, total_cost, cost_lookup, task_start_mask, task_goal_mask, agent_start_cost_tensor, agent_task_sequence_time
