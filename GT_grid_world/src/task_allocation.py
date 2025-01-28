from .analysis.statistics import Stats
from .graph import Graph
from .agent import *
from .utils import *

from .task_allocation_algorithms.closest_robot import closest_robot
from .task_allocation_algorithms.random import random
from .task_allocation_algorithms.cost_matrix import cost_matrix_TA
from .task_allocation_algorithms.lns import lns_call
from .task_allocation_algorithms.lns_fully_informed import lns_fi

import random

def TaskAllocation(S : Stats, G : Graph, Rs : AgentLoader, J : set, task_assignment_strategy : str, task_sequences : bool, map : str, hash_map : dict = {}) -> AgentLoader:
    """ Task allocation entrance function, which calls the respsective task assignment algorithm and returns the updated task assignment, set of free_agents, and set of to_pickup agents.

    Args:
        S (Stats): Statistics object
        G (Graph): Map data object
        Rs (list): Agent Loader Object
        J (set): Set of tasks in the form {(task_id, start_loc, goal_loc)}
        task_assignment_strategy (str): Chosen task_allocation algorithm to use
        to_pickup (set): Set of all robot_ids which are on the to_pickup segment of their task
        free_agents (set): Set of all robot_ids which are free agents
        hash_map (dict, optional): _description_. Defaults to {}.

    Returns:
        Tuple[list, set, set]: Returns updated task assignment, set of free_agents, and set of to_pickup agents
    """

    if task_assignment_strategy == "closest_robot":
        return closest_robot(S, G, Rs, Ra, J, task_assignment_strategy, to_pickup, free_agents, hash_map)
    elif task_assignment_strategy == "random":
        return random(S, Rs, Ra, J, to_pickup, free_agents)
    elif task_assignment_strategy == "lns":
        return lns_call(S, G, map, Rs, J)
    elif task_assignment_strategy == "lns_fully_informed":
        return lns_fi(S, G, map, Rs, J)
    elif task_assignment_strategy == "cost_matrix": 
        return cost_matrix_TA(S, G, Rs, J)
    else:
        print("ERROR: Unknown task assignment strategy " + task_assignment_strategy + ", please choose another one.")
        return Rs
    