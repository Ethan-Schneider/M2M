from .analysis.statistics import Stats
from .graph import Graph
from .agent import *
from .utils import *
from typing import Dict, Tuple
from .task_allocation_algorithms.c_lns import lns_call
from .task_allocation_algorithms.initial_solutions.max_regret_FC import max_regret_FC_call
from .task_allocation_algorithms.initial_solutions.randomized_max_regret_FC import randomized_max_regret_FC_call
from .task_allocation_algorithms.initial_solutions.FCF import FCF_call
from .task_allocation_algorithms.initial_solutions.greedy_allocation import greedy_call
from .task_allocation_algorithms.initial_solutions.randomized_greedy import randomized_greedy_call
from .task_allocation_algorithms.initial_solutions.random_allocation import random_call  
from .task_allocation_algorithms.py_lns_V2 import py_lns_call
from .task_allocation_algorithms.initial_solutions.fast_FCF import fast_FCF_call
from .task_allocation_algorithms.initial_solutions.fast_SCF import fast_SCF_call
from .task_allocation_algorithms.initial_solutions.fast_greedy import fast_greedy_call
from .task_allocation_algorithms.initial_solutions.construct_cost_elements import (
    CRM2M_DEFAULT_LAMBDA,
    CRM2M_DEFAULT_DETOUR_CUTOFF,
)
from .task_allocation_algorithms.hbh_mla_star import hbh_mla_star_call
from .task_allocation_algorithms.ta_hybrid_driver import ta_hybrid_call
from .task_allocation_algorithms.dual_cycle_allocation import apply_dual_cycle_pairing

def TaskAllocation(S : Stats, G : Graph, Rs : AgentLoader, J : Dict[int, Tuple], initial_task_assignment_strategy : str, 
                   improvement_task_assignment_strategy : str, map : str, t : int, cost_calculation_method : str, 
                   removal_operator : str = "worst", repair_operator : str = "greedy", 
                   acceptance_function: str = "greedy", T_0: float = 1.0, alpha: float = 0.99,
                   base_cost_weight: float = 1.0,
                   deadline_weight: float = 0.0,
                   sku_distribution_weight: float = 0.0,
                   agent_unallocated_penalty: float = 0.0,
                   schedule=None,
                   aisle_dual_cycle: bool = False,
                   driveway_dual_cycle: bool = False,
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

    # Dispatch to the chosen allocator. The variable ``result`` captures
    # the inner allocator's return so the dual-cycle post-pass below sees
    # exactly one allocator's output regardless of which branch ran. Some
    # allocators return ``AgentLoader`` directly; others (the fast_*/lns_*
    # families that ultimately wrap ``fast_greedy_allocation``) return a
    # 3-tuple ``(Rs, allocations, cost)`` -- we normalise to ``Rs``.
    result = None

    if improvement_task_assignment_strategy == "py_lns":
        result = py_lns_call(S, G, Rs, J, initial_task_assignment_strategy, time_limit=1.0, removal_size=2, cost_calculation_method=cost_calculation_method, removal_operator=removal_operator, repair_operator=repair_operator, t=t, acceptance_function=acceptance_function, T_0=T_0, alpha=alpha, base_cost_weight=base_cost_weight, deadline_weight=deadline_weight, sku_distribution_weight=sku_distribution_weight, agent_unallocated_penalty=agent_unallocated_penalty, crm2m_lambda=crm2m_lambda, crm2m_detour_cutoff=crm2m_detour_cutoff, crm2m_slack=crm2m_slack, crm2m_cpp=crm2m_cpp, crm2m_return_margin=crm2m_return_margin)
    elif improvement_task_assignment_strategy == "c_lns":
        result = lns_call(S, G, map, Rs, J, t)
    elif improvement_task_assignment_strategy == "hbh_mla_star":
        # Coupled allocator + path planner per Grenouilleau, van Hoeve, Hooker
        # (ICAPS 2019, pp. 181-185). HBH performs the assignment by calling
        # MLA* per (agent, task) candidate; the same MLA* call writes the
        # full path into agent.path_sequence so the external ECBS/PBS routing
        # step is skipped in execute() when this strategy is selected.
        result = hbh_mla_star_call(S, G, Rs, J, t)
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
        result = ta_hybrid_call(S, G, Rs, J, t, schedule=schedule)
    elif improvement_task_assignment_strategy == "none":
        # Falls through to the initial-allocator dispatch below.
        pass
    else:
        print("ERROR: Unknown task assignment strategy " + improvement_task_assignment_strategy + ", please choose another one.")
        return Rs

    if result is None:
        if initial_task_assignment_strategy == "random":
            result = random_call(S, G, Rs, J, method=cost_calculation_method)
        elif initial_task_assignment_strategy == "c_lns":
            result = lns_call(S, G, map, Rs, J, t)
        elif initial_task_assignment_strategy == "greedy":
            result = greedy_call(S, G, Rs, J, method=cost_calculation_method)
        elif initial_task_assignment_strategy == "randomized_greedy":
            result = randomized_greedy_call(S, G, Rs, J, method=cost_calculation_method)
        elif initial_task_assignment_strategy == "FCF":
            result = FCF_call(S, G, Rs, J, method=cost_calculation_method)
        elif initial_task_assignment_strategy == "max_regret_FC":
            result = max_regret_FC_call(S, G, Rs, J, method=cost_calculation_method)
        elif initial_task_assignment_strategy == "randomized_max_regret_FC":
            result = randomized_max_regret_FC_call(S, G, Rs, J, method=cost_calculation_method)
        elif initial_task_assignment_strategy == "fast_FCF":
            result = fast_FCF_call(S, G, Rs, J, t, method=cost_calculation_method)
        elif initial_task_assignment_strategy == "fast_SCF":
            result = fast_SCF_call(S, G, Rs, J, t, method=cost_calculation_method)
        elif initial_task_assignment_strategy == "fast_greedy":
            result = fast_greedy_call(S, G, Rs, J, t, method=cost_calculation_method, base_cost_weight=base_cost_weight, deadline_weight=deadline_weight, sku_distribution_weight=sku_distribution_weight, agent_task_sequence_limit=3, crm2m_lambda=crm2m_lambda, crm2m_detour_cutoff=crm2m_detour_cutoff, crm2m_slack=crm2m_slack, crm2m_cpp=crm2m_cpp, crm2m_return_margin=crm2m_return_margin)
        else:
            print("ERROR: Unknown task assignment strategy " + initial_task_assignment_strategy + ", please choose another one.")
            return Rs

    # Dual cycle pairing pipeline stage: pair IB+same-aisle-OB and
    # OB+same-driveway-IB into chained ``task_sequence`` tails. Mirrors
    # Star-Sim's pattern of handling dual cycles inside the allocation
    # pipeline (vs the previous reactive simulate-side hook), so the
    # chained tails participate in subsequent path planning and so the
    # eventual rearrangement-aware allocator can compose with them. The
    # stage re-runs every tick because ``fast_*`` and ``*_lns`` allocators
    # wipe trailing tasks on entry. ``apply_dual_cycle_pairing`` is a no-op
    # when both flags are off.
    if aisle_dual_cycle or driveway_dual_cycle:
        # ``hbh_mla_star`` and ``ta_hybrid`` write paths into
        # ``agent.path_sequence`` themselves; appending a chained task to
        # ``task_sequence`` here would not get a path planned for it on
        # this tick, and the simulator would hang the agent. Defer their
        # dual-cycling to a future MAPF-aware pairing pass; for now the
        # CLI guard in ``main()`` rejects this combination for ta_hybrid,
        # and hbh_mla_star ignores the request silently.
        if improvement_task_assignment_strategy not in ("hbh_mla_star", "ta_hybrid"):
            apply_dual_cycle_pairing(
                S, G,
                _normalise_to_rs(result),
                J,
                t,
                aisle_dual_cycle=aisle_dual_cycle,
                driveway_dual_cycle=driveway_dual_cycle,
            )

    return result


def _normalise_to_rs(result):
    """Return the ``AgentLoader`` from an allocator's return value.

    ``c_lns_call`` and a few allocators return the loader directly;
    ``fast_*`` and ``*_lns_call`` return a 3-tuple
    ``(Rs, allocations, cost)``. The dual-cycle pairing stage only needs
    the loader, so this helper centralises the unwrap.
    """
    if isinstance(result, tuple):
        return result[0]
    return result
    