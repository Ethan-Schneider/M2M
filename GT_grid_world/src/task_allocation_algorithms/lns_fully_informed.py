import numpy as np

from ..agent import *
from ..utils import *
from ..analysis.statistics import Stats
from ..graph import Graph

from ..task_allocation_algorithms.external_algorithms.lns_fully_informed import lns

def lns_fi(S : Stats, G : Graph, map_name : str, Rs : AgentLoader, J : set):
    
    map_name = "GT_grid_world/src/task_allocation_algorithms/external_algorithms/lns/maps/symbotic_small.map"

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
        
    # print("Unassigned Tasks:", unassigned_tasks)
    
    assigned_tasks_lns = []
    for agent in Rs.agents:
        if agent.task_sequence != []:
            assigned_tasks_lns.append((agent.id, get_task_start_location(J, agent.task_sequence[-1]), get_task_goal_location(J, agent.task_sequence[-1])))
    
    # print("Assigned Tasks: ", assigned_tasks_lns)
    
    # print("Agent States: ", Rs.get_agent_states())
    print(f"Agent Task Sequences After Purge: {[agent.task_sequence for agent in Rs.agents]}")
    Rs_final_states = []
    for robot in Rs.agents:
        Rs_final_states.append((robot.id, robot.state))
            
    sequences = [[] for _ in Rs.get_free_agents()]
    
    returned_sequence = lns.LNS(map_name, assigned_tasks_lns, unassigned_tasks, Rs_final_states, sequences)

    for robot_id, sequence in returned_sequence:
        if not sequence:
            continue
        if Rs.agents[robot_id-1].task_sequence == []:
            Rs.agents[robot_id-1].status = 1
            
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
    return Rs