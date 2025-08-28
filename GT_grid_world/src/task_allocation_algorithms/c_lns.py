import numpy as np

from typing import Dict, Tuple

from ..agent import *
from ..utils import *
from ..analysis.statistics import Stats
from ..graph import Graph

from ..task_allocation_algorithms.external_algorithms.lns import lns

def lns_call(S : Stats, G : Graph, map_name : str, Rs : AgentLoader, J : Dict[int, Tuple], t : int) -> AgentLoader:
    # map_name = "GT_grid_world/src/task_allocation_algorithms/external_algorithms/lns/maps/symbotic_small.map"
    map_name = "GT_grid_world/src/task_allocation_algorithms/external_algorithms/lns/maps/symbotic_medium_wide_deck.map"
    # Remove task sequence for each agent and add them to unassigned tasks
    for agent in Rs.agents:
        while len(agent.task_sequence) > 1:
            agent.task_sequence.pop(-1)

    assigned_tasks = Rs.get_all_assigned_tasks()
    all_task_ids = [key for key in J.keys()]
    assigned_task_ids = [task[0] for task in assigned_tasks]
    
    unassigned_task_ids = list(set(all_task_ids) - set(assigned_task_ids))
    unassigned_tasks = []
    
    assigned_locations = set()
    for agent in Rs.agents:
        for task in agent.task_sequence:
            assigned_locations.add(task[1])
            assigned_locations.add(task[2])

    warehouse_occupied_locs = set(G.warehouse.get_full_locations())
    driveway_occupied_locs = set(G.driveway.get_full_locations())

    unusable_locs = assigned_locations | warehouse_occupied_locs | driveway_occupied_locs

    empty_locs = set(G.driveway.get_empty_locations()) | set(G.warehouse.get_empty_locations())

    start_loc_unusable_locs = assigned_locations | empty_locs

    for task_id in unassigned_task_ids:
        # Find min distance between task's start and goal locations while removing any already assigned locations
        start_locations = list(set(J[task_id][0]) - set(start_loc_unusable_locs))
        goal_locations = list(set(J[task_id][1]) - set(unusable_locs))

        if start_locations and goal_locations:
            # Find the pair of start and goal locations with the minimum distance
            min_distance = float('inf')
            chosen_start_location = None
            chosen_goal_location = None
            for start_loc in start_locations:
                for goal_loc in goal_locations:
                    distance = G.get_distance(start_loc, goal_loc)
                    if distance < min_distance:
                        min_distance = distance
                        chosen_start_location = start_loc
                        chosen_goal_location = goal_loc

            if chosen_start_location is None or chosen_goal_location is None:
                continue
            else:
                unassigned_tasks.append((task_id, tuple(chosen_start_location), tuple(chosen_goal_location)))
                assigned_locations.add(chosen_start_location)
                assigned_locations.add(chosen_goal_location)

                start_loc_unusable_locs.add(chosen_start_location)
                start_loc_unusable_locs.add(chosen_goal_location)
                unusable_locs.add(chosen_start_location)
                unusable_locs.add(chosen_goal_location)
        else:
            continue
    
    # Create dict with keys of the task_id and values is the tuple of (start_loc, goal_loc)
    unassigned_tasks_dict = {task[0]: (task[1], task[2]) for task in unassigned_tasks}
    
    Rs_final_states = []
    for robot in Rs.agents:
        if robot.task_sequence == []:
            Rs_final_states.append((robot.id, robot.home))
            continue
        else:
            if robot.status == 1:
                Rs_final_states.append((robot.id, robot.task_sequence[-1][1]))
            elif robot.status == 2:
                Rs_final_states.append((robot.id, robot.task_sequence[-1][2]))
            else:
                print(f"Robot {robot.id} has an unknown status: {robot.status}")
                Rs_final_states.append((robot.id, robot.state))
            
    sequences = [[] for _ in Rs.agents]
    
    locations = []
    for task in unassigned_tasks:
        locations.append(task[1])
        locations.append(task[2])
    
    print(f"Unassigned tasks: {unassigned_tasks}")
    returned_sequence = lns.LNS(map_name, unassigned_tasks, Rs_final_states, sequences)
    
    assigned_returned_sequence = [x[1][0] for x in returned_sequence if x[1] != []]
    for task_id in unassigned_task_ids:
        if task_id in assigned_returned_sequence:
            S.add_task_reallocation(task_id)
    for robot_id, sequence in returned_sequence:
        if not sequence:
            continue
        for task_id in sequence:
            
            start_loc = unassigned_tasks_dict[task_id][0]
            goal_loc = unassigned_tasks_dict[task_id][1]
            if Rs.agents[robot_id-1].task_sequence == []:
                Rs.agents[robot_id-1].status = 1
                
            S.append_early_task_ids(task_id)
            
            Rs.agents[robot_id-1].task_sequence.append((task_id, tuple(start_loc), tuple(goal_loc), 9999))

            S.add_actual_distance(task_id)
            S.add_actual_pickup_distance(task_id)
            
            S.add_actual_duration(task_id)
            S.add_actual_pickup_duration(task_id)
            
    print(f"Agent task sequences after LNS: {[agent.task_sequence for agent in Rs.agents]}")
        
    return Rs, [], 0.0