from .analysis.statistics import Stats
from .graph import Graph
from .agent import *
from .utils import *

from .task_allocation_algorithms.lns import lns_call
from .task_allocation_algorithms.p_lns import p_lns_call
from .task_allocation_algorithms.initial_solutions.max_regret import max_regret_call
from .task_allocation_algorithms.initial_solutions.FCF import FCF_call
from .task_allocation_algorithms.initial_solutions.greedy_allocation import greedy_call
from .task_allocation_algorithms.initial_solutions.randomized_greedy import randomized_greedy_call
from .task_allocation_algorithms.initial_solutions.random_allocation import random_call  

def TaskAllocation(S : Stats, G : Graph, Rs : AgentLoader, J : set, task_assignment_strategy : str, map : str, t : int) -> AgentLoader:
    """ Task allocation entrance function, which calls the respsective task assignment algorithm and returns the updated task assignment, set of free_agents, and set of to_pickup agents.

    Args:
        S (Stats): Statistics object
        G (Graph): Map data object
        Rs (list): Agent Loader Object
        J (set): Set of tasks in the form {(task_id, start_loc, goal_loc)}
        task_assignment_strategy (str): Chosen task_allocation algorithm to use
        map (str): Map name
        t (int): Current timestep

    Returns:
        AgentLoader: Returns updated AgentLoader object
    """

    if task_assignment_strategy == "random":
        return random_call(S, G, Rs, J)
    elif task_assignment_strategy == "lns":
        return lns_call(S, G, map, Rs, J, t)
    elif task_assignment_strategy == "p_lns":
        return p_lns_call(S, G, map, Rs, J, t)
    elif task_assignment_strategy == "greedy":
        return greedy_call(S, G, Rs, J)
    elif task_assignment_strategy == "randomized_greedy":
        return randomized_greedy_call(S, G, Rs, J)
    elif task_assignment_strategy == "FCF":
        return FCF_call(S, G, Rs, J)
    elif task_assignment_strategy == "max_regret":
        return max_regret_call(S, G, Rs, J)
    else:
        print("ERROR: Unknown task assignment strategy " + task_assignment_strategy + ", please choose another one.")
        return Rs
    