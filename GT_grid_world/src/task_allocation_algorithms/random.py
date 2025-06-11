import numpy as np
from typing import Tuple

from ..analysis.statistics import Stats
from ..utils import *
from ..agent import *
from ..graph import Graph

def random_ta(G : Graph, S : Stats, Rs : AgentLoader, J : set) -> AgentLoader:
    """Randomly assign tasks to robots

    Args:
        S (Stats): Object for statistics
        Rs (AgentLoader): Agent Loader Object
        J (set): Set of tasks in the form {(task_id, start_loc, goal_loc)}

    Returns:
        AgentLoader: Updated Agent Loader Object
    """
    task_ids = set([x[0] for x in J])
    assigned_task_ids = Rs.get_all_assigned_tasks()
    unassigned_task_ids = task_ids - set(assigned_task_ids)
    
    unassigned_robot_ids = [agent.id for agent in Rs.get_free_agents()]
    
    # If there are no unassigned robots or tasks, return
    if len(unassigned_robot_ids) <= 0:
        return Rs
    elif len(unassigned_task_ids) <= 0:
        return Rs
    
    # Iterate through unassigned tasks and randomly assign to unassigned robots
    for task_id in unassigned_task_ids:
        robot_id = np.random.choice(unassigned_robot_ids)
        
        print(task_id)
        
        # Update Robot Task Sequence and Status
        Rs.agents[robot_id].task_sequence.append(task_id)
        Rs.agents[robot_id].status = 1
        
        # Update Statistics
        S.add_actual_distance(task_id)
        S.add_actual_pickup_distance(task_id)
        S.add_actual_duration(task_id)
        S.add_actual_pickup_duration(task_id)
        
        estimated_to_pickup_path = a_star(G, Rs.agents[robot_id].state, get_task_start_location(J, task_id))
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
    
    return Rs