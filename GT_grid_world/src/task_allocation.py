from typing import Tuple

from .analysis.statistics import Stats
from .graph import Graph
from .agent import Agent, AgentLoader
from .utils import *

from .task_allocation_algorithms.closest_robot import closest_robot
from .task_allocation_algorithms.random import random
from .task_allocation_algorithms.cost_matrix import cost_matrix_TA
from .task_allocation_algorithms.external_algorithms.lns import lns
import random

def TaskAllocation(S : Stats, G : Graph, Rs : AgentLoader, J : set, task_assignment_strategy : str, task_sequences : bool, to_pickup : set, map : str, hash_map : dict = {}) -> Tuple[list, set, set]:
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
        map_name = "GT_grid_world/src/task_allocation_algorithms/external_algorithms/lns/maps/symbotic_small.map"
        # map_name = ".src/task_allocation_algorithms/external_algorithms/lns/maps/" + map.split('/')[-1] + ".map"
        print(map_name)
    
        assigned_tasks = Rs.get_all_assigned_tasks()
        
        print(assigned_tasks)
    
    
        assigned_tasks = [(task_id, start_loc, goal_loc) for task_id, start_loc, goal_loc in J if task_id in assigned_tasks]
        unassigned_tasks = [(task_id, start_loc, goal_loc) for task_id, start_loc, goal_loc in J if task_id not in assigned_tasks]
        
        print(assigned_tasks)
        print(unassigned_tasks)
        
        Rs_final_states = []
        for robot in Rs.agents:
            if not robot.task_sequence:
                Rs_final_states.append((robot.id, robot.state))
            else:
                Rs_final_states.append((robot.id, robot.task_sequence[-1][2]))
                
        assigned_tasks = [(11, (9, 12), (20, 12))]
        unassigned_tasks = [(3, (9, 10), (20, 10)), (2, (2, 34), (19, 22)), (0, (4, 12), (20, 16)), (7, (18, 10), (13, 28)), (8, (20, 26), (12, 10)), (9, (20, 38), (5, 18)), (4, (20, 24), (12, 4)), (6, (20, 24), (5, 32)), (5, (7, 12), (18, 18)), (1, (19, 28), (3, 38))]
        sequences = [[] for _ in Rs.get_free_agents()]
        returned_sequence = lns.LNS(map_name, assigned_tasks, unassigned_tasks, Rs_final_states, sequences)
        
        for i, sequence in enumerate(returned_sequence):
            for task in sequence[-1]:
                Rs.agents[i].task_sequence.append(task)
            Rs.agents[i].free_agent = False
        
        assigned_tasks_set = set()
        for task_pair in returned_sequence:
            for task in task_pair[-1]:
                assigned_tasks_set.add(task)
        to_pickup.update(assigned_tasks_set)
            
        return Rs, to_pickup
            
    elif task_assignment_strategy == "cost_matrix": 
        return cost_matrix_TA(S, G, Rs, Ra, J, to_pickup, free_agents)
    