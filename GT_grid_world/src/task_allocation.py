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

def TaskAllocation(S : Stats, G : Graph, Rs : AgentLoader, J : set, task_assignment_strategy : str, task_sequences : bool, map : str, hash_map : dict = {}) -> Tuple[list, set, set]:
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
        
        print("Assigned Tasks: ", assigned_tasks)
    
        # COMPUTING THIS WRONG
        unassigned_tasks = [(task_id, start_loc, goal_loc) for task_id, start_loc, goal_loc in J if task_id not in assigned_tasks]
    
        print("Unassigned Tasks", unassigned_tasks)
        
        Rs_final_states = []
        for robot in Rs.agents:
            if robot.task_sequence == []:
                Rs_final_states.append((robot.id, robot.state))
            else:
                Rs_final_states.append((robot.id, get_task_goal_location(J, robot.task_sequence[-1])))
                
        sequences = [[] for _ in Rs.get_free_agents()]
        returned_sequence = lns.LNS(map_name, unassigned_tasks, Rs_final_states, sequences)
        
        print(returned_sequence)
        
        for robot_id, sequence in returned_sequence:
            if not sequence:
                continue
            for task_id in sequence:
                if task_id not in Rs.agents[robot_id].task_sequence:
                    Rs.agents[robot_id].task_sequence.append(task_id)
                    S.add_actual_distance(task_id)
                    S.add_actual_pickup_distance(task_id)
                    
                    S.add_actual_duration(task_id)
                    S.add_actual_pickup_duration(task_id)
            Rs.agents[robot_id].status = 1
            
        return Rs
            
    elif task_assignment_strategy == "cost_matrix": 
        return cost_matrix_TA(S, G, Rs, Ra, J, to_pickup, free_agents)
    