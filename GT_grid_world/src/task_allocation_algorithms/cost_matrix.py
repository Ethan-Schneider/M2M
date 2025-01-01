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
    list_ver = list(unassigned_task_ids)
    
    # If there are no free agents, skip task allocation
    if len(free_agents) <= 0:
        return Ra, to_pickup, free_agents
    # If there are no unassigned tasks, skip task allocation
    elif len(unassigned_task_ids) == 0:
        return Ra, to_pickup, free_agents

    
    # =============== Construct cost matrix ===============
    cost_matrix = []
    for unassigned_robot_id in free_agents:
        robot_cost = []
        for unassigned_task_id in unassigned_task_ids:
            # Compute path for agent-task pair
            path = a_star(G, get_robot_state(Rs, unassigned_robot_id), get_task_start_location(J, unassigned_task_id))
            
            # If cannot find path, set path length to infinite
            if not path:
                robot_cost.append(np.inf)
            # Else, attach the path length as the cost for the task-allocation pair
            else:
                robot_cost.append(len(path))
        # Append cost vector for robot i with all tasks
        cost_matrix.append(robot_cost)
    
    # =============== Edit Cost Matrix: Limit Number of Robots per Aisle and Driveway ===============
    
    # Construct list of current task locations in the driveway and aisle
    driveway_locations = []
    aisle_locations = []
    # Iterate over every currently allocated task
    for allocation in Ra:
        # Get the start and goal locations for the task
        start_loc = get_task_start_location(J, allocation[0])
        goal_loc = get_task_goal_location(J, allocation[0])
        
        # Check which position is in the driveway and aisle, and append them accordingly
        if start_loc[0] >= 16:
            driveway_locations.append(start_loc[1])
            aisle_locations.append(goal_loc[1])
        else:
            driveway_locations.append(goal_loc[1])
            aisle_locations.append(start_loc[1])
    
    aisle_locations = np.asarray(aisle_locations)
    driveway_locations = np.asarray(driveway_locations)
    cost_matrix = np.asarray(cost_matrix, dtype=float)   
    # Iterate over each unassigned task id to see if its aisle locations are already allocated too 
    for unassigned_task_id in unassigned_task_ids:
        # Get start and goal location for the task
        start_loc = get_task_start_location(J, unassigned_task_id)
        goal_loc = get_task_goal_location(J, unassigned_task_id)
        
        # Check whether the start or goal is for the aisle or driveway
        if start_loc[0] >= 16:
            aisle_loc = goal_loc[1]
            driveway_loc = start_loc[1]
        else:
            aisle_loc = start_loc[1]
            driveway_loc = goal_loc[1]
        
        # Check if aisle_loc or goal loc is already allocated twice

        if np.count_nonzero(driveway_locations == driveway_loc) >= 2:
            cost_matrix[:, list_ver.index(unassigned_task_id)] = np.inf
            continue
        
        if np.count_nonzero(aisle_locations == aisle_loc) >= 2:
            cost_matrix[:, list_ver.index(unassigned_task_id)] = np.inf
            continue
        
    # Exit if cost matrix is inf
    if np.all(np.isinf(cost_matrix)):
        return Ra, to_pickup, free_agents
    
    print("Cost Matrix Shape: ", cost_matrix.shape)
    
    # Iterate over cost matrix and assign agents to tasks
    is_solution = True
    try: 
        # TODO: Change to iterate over min of cost_matrix (row, col)
        for unassigned_robot_id in free_agents:
            # Get agent task allocation
            min_index_row, min_index_col = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            # Add allocation and statistics
            Ra.append((list(unassigned_task_ids)[min_index_col], list(free_agents)[min_index_row]))
            S.add_estimated_pickup_distance(list(unassigned_task_ids)[min_index_col], np.min(cost_matrix) - 1)
            S.add_estimated_pickup_duration(list(unassigned_task_ids)[min_index_col], np.min(cost_matrix) - 1)
            
            robot_id = list(free_agents)[min_index_row]
            free_agents.remove(robot_id)
            to_pickup.add(robot_id)
            
            S.add_actual_distance(get_assigned_task_id(Ra, robot_id))
            S.add_actual_pickup_distance(get_assigned_task_id(Ra, robot_id))
            
            S.add_actual_duration(get_assigned_task_id(Ra, robot_id))
            S.add_actual_pickup_duration(get_assigned_task_id(Ra, robot_id))
            
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
    
    return Ra, to_pickup, free_agents