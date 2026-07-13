from .analysis.statistics import Stats
from .graph import Graph
from .agent import *
from .utils import *
from typing import Dict, Tuple
from .task_allocation_algorithms.c_lns import lns_call
from .task_allocation_algorithms.M2M import M2M_call
from .task_allocation_algorithms.initial_solutions.fast_FCF import fast_FCF_call
from .task_allocation_algorithms.initial_solutions.fast_SCF import fast_SCF_call
from .task_allocation_algorithms.initial_solutions.fast_greedy import fast_greedy_call
from .task_allocation_algorithms.initial_solutions.construct_cost_elements import (
    CRM2M_DEFAULT_LAMBDA,
    CRM2M_DEFAULT_DETOUR_CUTOFF,
)
from .task_allocation_algorithms.hbh_mla_star import hbh_mla_star_call
from .task_allocation_algorithms.ta_hybrid_driver import ta_hybrid_call

def TaskAllocation(S : Stats, G : Graph, Rs : AgentLoader, J : Dict[int, Tuple], initial_task_assignment_strategy : str, 
                   improvement_task_assignment_strategy : str, map : str, t : int, cost_calculation_method : str, 
                   removal_operator : str = "worst", repair_operator : str = "greedy", 
                   acceptance_function: str = "greedy", T_0: float = 1.0, alpha: float = 0.99,
                   base_cost_weight: float = 1.0,
                   deadline_weight: float = 0.0,
                   sku_distribution_weight: float = 0.0,
                   agent_unallocated_penalty: float = 0.0,
                   schedule=None,
                   crm2m_lambda: float = CRM2M_DEFAULT_LAMBDA,
                   crm2m_detour_cutoff: float = CRM2M_DEFAULT_DETOUR_CUTOFF,
                   crm2m_slack: float = 0.0,
                   crm2m_cpp: float = 0.0,
                   crm2m_return_margin: float = 0.0) -> AgentLoader:
    """ Task allocation entrance function, which calls the respsective task assignment algorithm and returns the updated task assignment, set of free_agents, and set of to_pickup agents.

    Args:
        S (Stats): Statistics object
        G (Graph): Map data object
        Rs (list): Agent Loader Object
        J (dict): Dict of tasks in the form {task_id: (start_loc, goal_loc, deadline, sku_id, inbound)}
        initial_task_assignment_strategy (str): Chosen task_allocation algorithm to use
        improvement_task_assignment_strategy (str): Chosen task_allocation algorithm to use
        map (str): Map name
        t (int): Current timestep
        cost_calculation_method (str): Method for calculating cost ("manhattan" or "shortest_path")
        removal_operator (str): Removal operator for LNS ("random" or "worst")
        repair_operator (str): Repair operator for LNS ("greedy")
        acceptance_function (str): Acceptance function for py_lns
        T_0 (float): Initial temperature for py_lns
        alpha (float): Cooling rate for py_lns

    Returns:
        AgentLoader: Returns updated AgentLoader object
    """

    # ``M2M`` (a.k.a. the legacy ``py_lns`` key) routes through the
    # renamed M2M module which now carries the crM2M rearrangement terms.
    if improvement_task_assignment_strategy in ("M2M", "py_lns"):
        return M2M_call(S, G, Rs, J, initial_task_assignment_strategy, time_limit=1.0, removal_size=2, cost_calculation_method=cost_calculation_method, removal_operator=removal_operator, repair_operator=repair_operator, t=t, acceptance_function=acceptance_function, T_0=T_0, alpha=alpha, base_cost_weight=base_cost_weight, deadline_weight=deadline_weight, sku_distribution_weight=sku_distribution_weight, agent_unallocated_penalty=agent_unallocated_penalty, crm2m_lambda=crm2m_lambda, crm2m_detour_cutoff=crm2m_detour_cutoff, crm2m_slack=crm2m_slack, crm2m_cpp=crm2m_cpp, crm2m_return_margin=crm2m_return_margin)
    elif improvement_task_assignment_strategy == "c_lns":
        return lns_call(S, G, map, Rs, J, t)
    elif improvement_task_assignment_strategy == "hbh_mla_star":
        # Coupled allocator + path planner per Grenouilleau, van Hoeve, Hooker
        # (ICAPS 2019, pp. 181-185). HBH performs the assignment by calling
        # MLA* per (agent, task) candidate; the same MLA* call writes the
        # full path into agent.path_sequence so the external ECBS/PBS routing
        # step is skipped in execute() when this strategy is selected.
        return hbh_mla_star_call(S, G, Rs, J, t)
    elif improvement_task_assignment_strategy == "ta_hybrid":
        # Offline TA-Hybrid (Liu, Ma, Li, Koenig, AAMAS 2019). The
        # driver runs a special-TSP task assignment once at t=0 and
        # then coordinates per-tick PlanPathsToDelivery (ICBS) and
        # PlanPathsToPickup (time-extended min-cost max-flow) calls.
        # Like HBH+MLA*, it's a coupled allocator + path planner: the
        # driver writes the full path into agent.path_sequence so the
        # external ECBS/PBS routing step is skipped in execute() when
        # this strategy is selected. Requires --use-precomputed-schedule
        # (the offline assumption).
        return ta_hybrid_call(S, G, Rs, J, t, schedule=schedule)
    elif improvement_task_assignment_strategy == "none":
        # Falls through to the initial-allocator dispatch below.
        pass
    else:
        print("ERROR: Unknown task assignment strategy " + improvement_task_assignment_strategy + ", please choose another one.")
        return Rs

    if initial_task_assignment_strategy == "c_lns":
        return lns_call(S, G, map, Rs, J, t)
    elif initial_task_assignment_strategy == "fast_FCF":
        return fast_FCF_call(S, G, Rs, J, t, method=cost_calculation_method)
    elif initial_task_assignment_strategy == "fast_SCF":
        return fast_SCF_call(S, G, Rs, J, t, method=cost_calculation_method)
    elif initial_task_assignment_strategy == "fast_greedy":
        return fast_greedy_call(S, G, Rs, J, t, method=cost_calculation_method, base_cost_weight=base_cost_weight, deadline_weight=deadline_weight, sku_distribution_weight=sku_distribution_weight, agent_task_sequence_limit=3, crm2m_lambda=crm2m_lambda, crm2m_detour_cutoff=crm2m_detour_cutoff, crm2m_slack=crm2m_slack, crm2m_cpp=crm2m_cpp, crm2m_return_margin=crm2m_return_margin)
    else:
        print("ERROR: Unknown task assignment strategy " + initial_task_assignment_strategy + ", please choose another one.")
        return Rs
    