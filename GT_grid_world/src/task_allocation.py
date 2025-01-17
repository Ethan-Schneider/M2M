from typing import Tuple

from .analysis.statistics import Stats
from .graph import Graph
from .utils import *

from .task_allocation_algorithms.closest_robot import closest_robot
from .task_allocation_algorithms.random import random
from .task_allocation_algorithms.cost_matrix import cost_matrix_TA
from .task_allocation_algorithms.external_algorithms.lns import lns
import random

def TaskAllocation(S : Stats, G : Graph, Rs : list, Ra : list, J : set, task_assignment_strategy : str, to_pickup : set, free_agents : set, map : str, sequences : list, hash_map : dict = {}) -> Tuple[list, set, set]:
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
    elif task_assignment_strategy == "lns":
        map_name = "GT_grid_world/src/task_allocation_algorithms/external_algorithms/lns/maps/symbotic_small.map"
        # map_name = ".src/task_allocation_algorithms/external_algorithms/lns/maps/" + map.split('/')[-1] + ".map"
        print(map_name)
        
        assigned_tasks = {task_id for task_id, _ in Ra}

        assigned_task_details = [(task_id, start_loc, goal_loc) for task_id, start_loc, goal_loc in J if task_id in assigned_tasks]
        unassigned_task_details = [(task_id, start_loc, goal_loc) for task_id, start_loc, goal_loc in J if task_id not in assigned_tasks]

        print("Assigned tasks:", assigned_task_details)
        print("Unassigned tasks:", unassigned_task_details)
        
        print("Agents:", Rs)
        for task_id, _, _ in unassigned_task_details:
            if free_agents:
                assigned_agent = random.choice(list(free_agents))
                Ra.append((task_id, assigned_agent))
                free_agents.remove(assigned_agent)
                to_pickup.add(assigned_agent)
                
                # Update sequences with sequences of tasks for each robot
                if assigned_agent not in sequences:
                    sequences[assigned_agent].append(task_id)
        print("Assigned tasks:", Ra)
        print("Sequences:", sequences)
        
        lns.LNS(map_name, assigned_task_details, unassigned_task_details, Rs, Ra)
    elif task_assignment_strategy == "cost_matrix": 
        return cost_matrix_TA(S, G, Rs, Ra, J, to_pickup, free_agents)
    