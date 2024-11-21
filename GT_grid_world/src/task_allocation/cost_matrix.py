import numpy as np
from typing import Tuple

from ..analysis.statistics import Stats
from ..graph import Graph
from ..utils import *

def cost_matrix_TA(S : Stats, G : Graph, Rs : list, Ra : list, J : set, to_pickup : set, free_agents : set) -> Tuple[list, set, set]:
    """A baseline task allocation algorithm, which constructs a cost matrix of the distance between each free agent and each unassigned task, then
    iteratively assign the minimum cost from the matrix until all tasks or robot have been assigned.

    Args:
        S (Stats): Statistics object
        G (Graph): Map data object
        Rs (list): Robot State in the form of [(robot_id, state), ...]
        Ra (list): Robot-Task Allocation in the form of [(task_id, robot_id), ...]
        J (set): Set of tasks in the form {(task_id, start_loc, goal_loc)}
        to_pickup (set): Set of all robot_ids which are on the to_pickup segment of their task
        free_agents (set): Set of all robot_ids which are free agents

    Returns:
        Ra(list), to_pickup(set), free_agents(set): Returns updated task assignment, set of free_agents, and set of to_pickup agents
    """
    
    # If no free agents or unassinged tasks, then return current allocation, to_pickup, and free_agents
    unassigned_task_ids = get_unassigned_task_ids(J, Ra)    
    
    if len(free_agents) <= 0:
        return Ra, to_pickup, free_agents
    elif len(unassigned_task_ids) <= 0:
        return Ra, to_pickup, free_agents
    else:
        pass
    
    # Construct cost matrix
    cost_matrix = []
    for unassigned_robot_id in free_agents:
        robot_cost = []
        for unassigned_task_id in unassigned_task_ids:
            # TODO: Debug a_star implementation for why this occurs so infrequently
            path = a_star(G, get_robot_state(Rs, unassigned_robot_id), get_task_start_location(J, unassigned_task_id))
            
            # If cannot find path, set path length to infinite
            if not path:
                robot_cost.append(np.inf)
            else:
                robot_cost.append(len(path))
        cost_matrix.append(robot_cost)
    
    # Iterate over cost matrix and assign agents to tasks
    is_solution = True
    cost_matrix = np.asarray(cost_matrix, dtype=float)   
    try: 
        # TODO: Change to iterate over min of cost_matrix (row, col)
        for unassigned_robot_id in free_agents:
            # Get agent task allocation
            min_index_row, min_index_col = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            # Add allocation and statistics
            Ra.append((list(unassigned_task_ids)[min_index_col], list(free_agents)[min_index_row]))
            S.add_estimated_pickup_distance(list(unassigned_task_ids)[min_index_col], np.min(cost_matrix) - 1)
            S.add_estimated_pickup_duration(list(unassigned_task_ids)[min_index_col], np.min(cost_matrix) - 1)
            
            # TODO: Find out why a_star is returning no solution occasionally 
            # Compute estimated distance from pickup to place 
            while True:
                estimated_task_path = a_star(G, get_task_start_location(J, list(unassigned_task_ids)[min_index_col]), get_task_goal_location(J, list(unassigned_task_ids)[min_index_col]))
                if estimated_task_path:
                    break
                elif not estimated_task_path:
                    print(get_task_start_location(J, list(unassigned_task_ids)[min_index_col])) 
                    print(get_task_goal_location(J, list(unassigned_task_ids)[min_index_col]))
                    is_solution = False
                    break
            if is_solution:
                S.add_estimated_distance(list(unassigned_task_ids)[min_index_col], len(estimated_task_path) - 1)
                S.add_estimated_duration(list(unassigned_task_ids)[min_index_col], len(estimated_task_path) - 1)
            else:
                S.add_estimated_distance(list(unassigned_task_ids)[min_index_col], np.Infinity)
                S.add_estimated_duration(list(unassigned_task_ids)[min_index_col], np.Infinity)
            is_solution = True
            
            # Update matrix values
            cost_matrix[:, min_index_col] = np.inf
            cost_matrix[min_index_row, :] = np.inf
            
            # Remove assigned task from unassigned_task_list
            unassigned_task_ids.remove(list(unassigned_task_ids)[min_index_col])
            if len(unassigned_task_ids) == 0:
                break
    except:
        return Ra, to_pickup, free_agents

    # Update statistics, free agents, and to_pickup
    for task in Ra:
        robot_id = task[-1]
        if robot_id in free_agents:
            free_agents.remove(robot_id)
            to_pickup.add(robot_id)
            
            S.add_actual_distance(get_assigned_task_id(Ra, robot_id))
            S.add_actual_pickup_distance(get_assigned_task_id(Ra, robot_id))
            
            S.add_actual_duration(get_assigned_task_id(Ra, robot_id))
            S.add_actual_pickup_duration(get_assigned_task_id(Ra, robot_id))
        else:
            continue
    return Ra, to_pickup, free_agents