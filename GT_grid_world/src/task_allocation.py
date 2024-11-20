from typing import Tuple

from .statistics import Stats
from .graph import Graph
from .utils import *

from task_allocation.closest_robot import closest_robot
from task_allocation.random import random
from task_allocation.cost_matrix import cost_matrix_TA

def TaskAllocation(S : Stats, G : Graph, Rs : list, Ra : list, J : set, task_assignment_strategy : str, to_pickup : set, free_agents : set, hash_map : dict = {}) -> Tuple[list, set, set]:
    """ Task allocation entrance function, which calls the respsective task assignment algorithm and returns the updated task assignment, set of free_agents, and set of to_pickup agents.

    Args:
        S (Stats): Statistics object
        G (Graph): Map data object
        Rs (list): Robot State in the form of [(robot_id, state), ...]
        Ra (list): Robot-Task Allocation in the form of [(task_id, robot_id), ...]
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
    
    elif task_assignment_strategy == "cost_matrix": 
        return cost_matrix_TA(S, G, Rs, Ra, J, to_pickup, free_agents, hash_map)
    