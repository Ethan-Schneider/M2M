import numpy as np
import time
from typing import List, Tuple, Dict
from ...agent import AgentLoader
from ...analysis.statistics import Stats
from ...graph import Graph
from ...utils import manhattan_distance
from ..initial_solutions.construct_cost_elements import (
    per_task_type_sku_distribution_term,
    compute_crm2m_terms,
    rearrangement_cost_cube,
    TASK_TYPE_SHUFFLE,
    CRM2M_DEFAULT_LAMBDA,
    CRM2M_DEFAULT_DETOUR_CUTOFF,
)

def fast_SCF_repair(S: Stats, G: Graph, agent_start_cost_tensor: np.ndarray, start_goal_dist: np.ndarray, task_start_mask: np.ndarray, task_goal_mask: np.ndarray, Rs: AgentLoader,
                 start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]],
                 idx_to_task_id: Dict[int, int], temp_allocations: List[Tuple[int, int, int, int]],
                 method: str = "manhattan", cost_lookup: Dict[Tuple[int, int, int, int], int] = None,
                 J: Dict[int, Tuple] = None, inbound_sku_distribution_costs: np.ndarray = None, 
                 outbound_sku_distribution_costs: np.ndarray = None,
                 rearrangement_sku_distribution_costs: np.ndarray = None,
                 base_cost_weight: float = 1.0, 
                 deadline_weight: float = 0.0, sku_distribution_weight: float = 0.0,
                 agent_task_sequence_time: np.ndarray = None, current_time: int = 0,
                 crm2m_lambda: float = CRM2M_DEFAULT_LAMBDA, crm2m_detour_cutoff: float = CRM2M_DEFAULT_DETOUR_CUTOFF) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], float]:
    """
    Repair a solution using Second Coordinate Fixing (SCF) algorithm.
    Iterates over tasks and assigns each task to the best available agent.
    
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
        cost_lookup: Dictionary to store allocation costs
        current_time: Current timestep for deadline calculations
        
    Returns:
        Tuple containing:
        - AgentLoader with updated task sequences
        - List of allocations (m, n, p, q)
        - Total cost of all allocations
    """
    total_cost = 0.0
    allocations = []
    M = len(Rs.agents)
    N = len(idx_to_task_id)
    P = len(start_locs)

    # Copy masks so we can update them
    task_start_mask_ = task_start_mask.copy()
    task_goal_mask_ = task_goal_mask.copy()

    # crM2M static terms (see fast_greedy_allocation). This operator scores by
    # *argmax* (higher == better), so type=2 uses +U with gated entries set to
    # -inf, the mirror of fast_greedy's -U / +inf convention. Gated on a type=2
    # task being present so plain-M2M ticks skip the dead computation.
    has_shuffle = any(
        J[task_id][4] == TASK_TYPE_SHUFFLE for task_id in idx_to_task_id.values()
    )
    if has_shuffle:
        (crm2m_home, crm2m_start_home, crm2m_goal_home,
         crm2m_agent_home, crm2m_coupling_mask) = compute_crm2m_terms(
            Rs, G, start_locs, goal_locs, method
        )
    else:
        crm2m_home = crm2m_start_home = crm2m_goal_home = None
        crm2m_agent_home = crm2m_coupling_mask = None

    # Iterate over all tasks (SCF approach)
    for n in range(N):
        # Skip if task is already assigned
        if idx_to_task_id[int(n)] in [key[1] for key in cost_lookup.keys()]:
            continue
            
        valid_p = np.where(task_start_mask_[n] == 1)[0]
        valid_q = np.where(task_goal_mask_[n] == 1)[0]
        if len(valid_p) == 0 or len(valid_q) == 0:
            continue

        task_type = J[idx_to_task_id[int(n)]][4]

        # crM2M: +U cube over (M, |valid_p|, |valid_q|), computed once per task.
        # ``agent_start_cost_tensor`` here holds NEGATED distances (this operator's
        # convention), so abs() recovers the dist(g^m_{i-1}, s_p) the detour needs.
        u_cube = None
        if task_type == TASK_TYPE_SHUFFLE:
            u_cube = -rearrangement_cost_cube(
                np.abs(agent_start_cost_tensor), crm2m_agent_home, crm2m_start_home,
                crm2m_goal_home, crm2m_coupling_mask, valid_p, valid_q,
                crm2m_lambda, crm2m_detour_cutoff,
            )

        # Find the best agent for this task (SCF: iterate over agents for each task)
        best_cost = -np.inf
        best = None
        
        for m in range(M):
            if task_type == TASK_TYPE_SHUFFLE:
                total_costs = u_cube[m]
            else:
                # Vectorized cost computation for all valid (p, q) pairs
                agent_costs = agent_start_cost_tensor[m, valid_p][:, None]
                sg_costs = start_goal_dist[np.ix_(valid_p, valid_q)]

                # Calculate base costs with deadline and agent_task_sequence_time considerations
                deadline = J[idx_to_task_id[int(n)]][2]
                if deadline_weight > 0.0:
                    # If deadline has not passed
                    if deadline - current_time > 0:
                        base_costs = -1*deadline_weight*(deadline - current_time) + base_cost_weight*((agent_costs + sg_costs) + agent_task_sequence_time[m])
                    # If deadline has passed
                    else:
                        base_costs = deadline_weight*np.abs(deadline - current_time) + base_cost_weight*((agent_costs + sg_costs) + agent_task_sequence_time[m])
                else:
                    base_costs = base_cost_weight*(agent_costs + sg_costs)

                # 1.6: per-task-type SKU-distribution placement quality (three-way dispatch).
                # NOTE: this allocator uses a negated-cost / argmax convention different
                # from fast_greedy's argmin convention, and its `deadline_weight` branch
                # above is on a different scale than `task_deadline_costs`. Unifying the
                # two is a follow-up cleanup; 1.6 only updates the SKU-distribution side
                # so the per-task-type matrices are routed consistently.
                sku_term, axis = per_task_type_sku_distribution_term(
                    task_type, n, valid_p, valid_q,
                    inbound_sku_distribution_costs=inbound_sku_distribution_costs,
                    outbound_sku_distribution_costs=outbound_sku_distribution_costs,
                    rearrangement_sku_distribution_costs=rearrangement_sku_distribution_costs,
                )
                if axis == "goals":
                    total_costs = base_costs + sku_distribution_weight * sku_term[None, :]
                else:
                    total_costs = base_costs + sku_distribution_weight * sku_term[:, None]

            max_idx = np.argmax(total_costs)
            max_cost_m = total_costs.flat[max_idx]
            if max_cost_m > best_cost:
                best_cost = max_cost_m
                p_idx, q_idx = np.unravel_index(max_idx, total_costs.shape)
                best = (m, valid_p[p_idx], valid_q[q_idx])
                
        if best is None or best_cost == -np.inf:
            continue

        # Add the task to the allocation
        m, p, q = best
        allocations.append((int(m), idx_to_task_id[int(n)], int(p), int(q)))
        total_cost += best_cost

        # Store cost in lookup table if provided
        if cost_lookup is not None:
            cost_lookup[(int(m), idx_to_task_id[int(n)], int(p), int(q))] = int(best_cost)

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
        task_start_mask_[:, p] = 0
        task_goal_mask_[:, q] = 0
        task_start_mask_[n, :] = 0
        task_goal_mask_[n, :] = 0

        # Update costs for agent-start allocation for agent m
        for p_ in range(P):
            if method == "manhattan":
                cost = -1.0 * manhattan_distance(goal_locs[q], start_locs[p_])
            elif method == "shortest_path":
                cost = -1.0 * G.get_distance(goal_locs[q], start_locs[p_])
            else:
                raise ValueError(f"Invalid cost calculation method: {method}")
            agent_start_cost_tensor[m, p_] = cost

        # crM2M: refresh agent m's distance to h_0 after its anchor moved to
        # goal_locs[q] (stored positive, matching compute_crm2m_terms). Skipped
        # when no shuffle is present (terms were never computed).
        if has_shuffle:
            if method == "manhattan":
                crm2m_agent_home[m] = manhattan_distance(goal_locs[q], crm2m_home)
            else:
                crm2m_agent_home[m] = G.get_distance(goal_locs[q], crm2m_home)

        # Update agent task sequence time
        if len(Rs.agents[m].task_sequence) > 1:
            agent_task_sequence_time[m] -= G.get_distance(Rs.agents[m].task_sequence[-2][2], Rs.agents[m].task_sequence[-1][1])
            agent_task_sequence_time[m] -= G.get_distance(Rs.agents[m].task_sequence[-1][1], Rs.agents[m].task_sequence[-1][2])
        else:
            agent_task_sequence_time[m] -= G.get_distance(Rs.agents[m].state, Rs.agents[m].task_sequence[0][1])
            agent_task_sequence_time[m] -= G.get_distance(Rs.agents[m].task_sequence[0][1], Rs.agents[m].task_sequence[0][2])

    return Rs, allocations, total_cost, cost_lookup, task_start_mask_, task_goal_mask_, agent_start_cost_tensor
