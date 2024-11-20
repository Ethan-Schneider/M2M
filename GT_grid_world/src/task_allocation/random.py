import numpy as np
from typing import Tuple

from ..statistics import Stats
from ..utils import *

def random(S : Stats, Rs : list, Ra : list, J : set, to_pickup : set, free_agents : set) -> Tuple[list, set, set]:
    """Randomly Assign Unallocated Tasks to Agents

    Args:
        S (Stats): Statistics Object
        Rs (list): Robot State in the form of [(robot_id, state), ...]
        Ra (list): Robot-Task Allocation in the form of [(task_id, robot_id), ...]
        J (set): Set of tasks in the form {(task_id, start_loc, goal_loc)}
        to_pickup (set): Set of all robot_ids which are on the to_pickup segment of their task
        free_agents (set): Set of all robot_ids which are free agents

    Returns:
        Ra (list), to_pickup (set), free_agents (set): Returns updated task assignment, set of free_agents, and set of to_pickup agents
    """
    assigned_task_ids = set([x[0] for x in Ra])
    task_ids = set([x[0] for x in J])
    
    unassigned_task_ids = task_ids - assigned_task_ids
    
    assigned_robot_ids = set()
    for assignment in Ra:
        assigned_robot_ids |= set([assignment[-1]])
    unassigned_robot_ids = set(np.arange(0, len(Rs))) - assigned_robot_ids
    
    if len(unassigned_robot_ids) <= 0:
        return Ra, to_pickup, free_agents
    elif len(unassigned_task_ids) <= 0:
        return Ra, to_pickup, free_agents
    else:
        pass
    
    for unassigned_robot_id in unassigned_robot_ids:
        task = np.random.choice(list(unassigned_task_ids))
        
        Ra.append((task, unassigned_robot_id))
        unassigned_task_ids.remove(task)
        if len(unassigned_task_ids) == 0:
            break
    
    
    for task in Ra:
        robot_id = task[-1]
        if robot_id in free_agents:
            free_agents.remove(robot_id)
            to_pickup.add(robot_id)
            
            S.add_actual_distance(get_assigned_task_id(Ra, robot_id))
            S.add_actual_pickup_distance(get_assigned_task_id(Ra, robot_id))
        else:
            continue
    
    return Ra, to_pickup, free_agents