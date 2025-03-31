import numpy as np

from ..agent import *
from ..utils import *
from ..analysis.statistics import Stats
from ..graph import Graph

from ..task_allocation_algorithms.external_algorithms.lns import lns
from ..task_allocation_algorithms.external_algorithms.lns import dc

def lns_call(S : Stats, G : Graph, map_name : str, Rs : AgentLoader, J : set, t : int) -> AgentLoader:

    map_name = "GT_grid_world/src/task_allocation_algorithms/external_algorithms/lns/maps/symbotic_small.map"
    # map_name = "GT_grid_world/src/task_allocation_algorithms/external_algorithms/lns/maps/symbotic_large.map"

    # for agent in Rs.agents:
    #     print("Pre Agent Task Sequence", agent.task_sequence)
    print(f"Agent Task Sequences Prior to Purge: {[agent.task_sequence for agent in Rs.agents]}")
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
    
    print(f"Agent Task Sequences After Purge: {[agent.task_sequence for agent in Rs.agents]}")
    Rs_final_states = []
    for robot in Rs.get_free_agents():
        if robot.task_sequence == []:
            Rs_final_states.append((robot.id, robot.state))
        else:
            Rs_final_states.append((robot.id, get_task_goal_location(J, robot.task_sequence[-1])))
            
    sequences = [[] for _ in Rs.get_free_agents()]
    
    print(f"Unassigned Task Ids: {unassigned_task_ids}")
    
    
    returned_sequence = lns.LNS(map_name, unassigned_tasks, Rs_final_states, sequences)
    print(f"Returned Sequence: {returned_sequence}")
    
    assigned_returned_sequence = [x[1][0] for x in returned_sequence if x[1] != []]
    for task_id in unassigned_task_ids:
        if task_id in assigned_returned_sequence:
            S.add_task_reallocation(task_id)

    for robot_id, sequence in returned_sequence:
        if not sequence:
            continue
        robot_id = Rs.get_free_agents()[robot_id-1].id + 1 # Get the actual robot id from the free agents list
        if Rs.agents[robot_id-1].task_sequence == []:
            Rs.agents[robot_id-1].status = 1
            task_id = sequence[0]
            
            S.append_early_task_ids(task_id)
            
        # for task_id in sequence:
        #     if task_id not in Rs.agents[robot_id-1].task_sequence:
            Rs.agents[robot_id-1].task_sequence.append(task_id)

            S.add_actual_distance(task_id)
            S.add_actual_pickup_distance(task_id)
            
            S.add_actual_duration(task_id)
            S.add_actual_pickup_duration(task_id)

            # print(f"Estimated Distance for {Rs.agents[robot_id-1].state} to {get_task_start_location(J, task_id)} {dc.distance(map_name, Rs.agents[robot_id-1].state, get_task_start_location(J, task_id))}")

            estimated_to_pickup_path = dc.distance(map_name, Rs.agents[robot_id-1].state, get_task_start_location(J, task_id))
            estimated_task_path = dc.distance(map_name, get_task_start_location(J, task_id), get_task_goal_location(J, task_id))
            
            print(f"Estimated Distance for task {task_id} start to pickup: {Rs.agents[robot_id-1].state} to {get_task_start_location(J, task_id)} {estimated_to_pickup_path}")
            print(f"Estimated Distance for task {task_id} start to pickup: {get_task_start_location(J, task_id)} to {get_task_goal_location(J, task_id)} {estimated_task_path}")
        

            S.add_estimated_pickup_duration(task_id, estimated_to_pickup_path)
            S.add_estimated_pickup_distance(task_id, estimated_to_pickup_path)

            S.add_estimated_distance(task_id, estimated_task_path)
            S.add_estimated_duration(task_id, estimated_task_path)
        else:
            for task_id in sequence:
                if task_id not in Rs.agents[robot_id-1].task_sequence:
                    Rs.agents[robot_id-1].task_sequence.append(task_id)
        
    return Rs