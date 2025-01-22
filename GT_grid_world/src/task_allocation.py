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

        # for agent in Rs.agents:
        #     print("Pre Agent Task Sequence", agent.task_sequence)
        # Remove task sequence for each agent and add them to unassigned tasks
        for agent in Rs.agents:
            while len(agent.task_sequence) > 1:
                agent.task_sequence.pop(-1)
        # print("=======================")
        # for agent in Rs.agents:
        #     print("Post Agent Task Sequence", agent.task_sequence)

        assigned_tasks = Rs.get_all_assigned_tasks()
        all_task_ids = [task[0] for task in J]
        
        unassigned_task_ids = list(set(all_task_ids) - set(assigned_tasks))
        unassigned_tasks = []

        for task_id in unassigned_task_ids:
            unassigned_tasks.append((task_id, get_task_start_location(J, task_id), get_task_goal_location(J, task_id)))
        
        Rs_final_states = []
        for robot in Rs.agents:
            if robot.task_sequence == []:
                Rs_final_states.append((robot.id, robot.state))
            else:
                Rs_final_states.append((robot.id, get_task_goal_location(J, robot.task_sequence[-1])))
                
        sequences = [[] for _ in Rs.get_free_agents()]
        returned_sequence = lns.LNS(map_name, unassigned_tasks, Rs_final_states, sequences)

        for robot_id, sequence in returned_sequence:
            if not sequence:
                continue
            for task_id in sequence:
                if task_id not in Rs.agents[robot_id-1].task_sequence:
                    Rs.agents[robot_id-1].task_sequence.append(task_id)
                    S.add_actual_distance(task_id)
                    S.add_actual_pickup_distance(task_id)
                    
                    S.add_actual_duration(task_id)
                    S.add_actual_pickup_duration(task_id)

                    estimated_to_pickup_path = a_star(G, Rs.agents[robot_id-1].state, get_task_start_location(J, task_id))
                    estimated_task_path = a_star(G, get_task_start_location(J, task_id), get_task_goal_location(J, task_id))

                    if not estimated_to_pickup_path:
                        S.add_estimated_pickup_distance(task_id, np.inf)
                        S.add_estimated_pickup_duration(task_id, np.inf)
                    else:
                        S.add_estimated_pickup_duration(task_id, len(estimated_to_pickup_path)-1)
                        S.add_estimated_pickup_distance(task_id, len(estimated_to_pickup_path)-1)

                    if not estimated_task_path:
                        S.add_estimated_distance(task_id, np.inf)
                        S.add_estimated_duration(task_id, np.inf)
                    else:
                        S.add_estimated_distance(task_id, len(estimated_task_path) - 1)
                        S.add_estimated_duration(task_id, len(estimated_task_path) - 1)
            Rs.agents[robot_id-1].status = 1
            
        return Rs
            
    elif task_assignment_strategy == "cost_matrix": 
        return cost_matrix_TA(S, G, Rs, Ra, J, to_pickup, free_agents)
    