from typing import Tuple

from .analysis.statistics import Stats
from .analysis.buffer import Buffer
from .graph import Graph
from .agent import *
from .utils import *

from .task_allocation_algorithms.external_algorithms.lns import dc


def simulate(S : Stats, B : Buffer, G : Graph, Rs : AgentLoader, J : set, map_name : str, t : int) -> Tuple[AgentLoader, set]:
    map_name = "GT_grid_world/src/task_allocation_algorithms/external_algorithms/lns/maps/symbotic_small.map"
    # Update state of robots
    for agent in Rs.agents:
        # If robot sequence is stationary, leave the robot in place (wait action)
        # BUG: crashes when router fails to find a solution 
        if len(agent.path_sequence) == 0:
            continue
        # If robot does have a sequence of actions, pop next state and update
        else:
            # Update Agent State and Graph Occupied States
            old_state = agent.state
            G.set_occupied(agent.state, False)
            agent.state = agent.path_sequence.pop(0)
            G.set_occupied(agent.state, True)
            
            if agent.status == 1:
                # if agent.task_sequence[0] == 2:
                #     print(f"Task 2 start location: {get_task_start_location(J, 2)} and goal location: {get_task_goal_location(J, 2)} with agent state: {agent.state}")
                #     print(euclidian_distance(old_state, agent.state))
                #     print(f"Duration so far for first part of task {agent.task_sequence[0]}: {S.get_actual_pickup_duration(agent.task_sequence[0])}")
                S.update_actual_pickup_distance(agent.task_sequence[0], euclidian_distance(old_state, agent.state))
                S.update_actual_pickup_duration(agent.task_sequence[0], 1)
            elif agent.status == 2:
                # if agent.task_sequence[0] == 2:
                #     print(euclidian_distance(old_state, agent.state))
                #     print(f"Duration so far for second part task {agent.task_sequence[0]}: {S.get_actual_duration(agent.task_sequence[0])}")
                S.update_actual_distance(agent.task_sequence[0], euclidian_distance(old_state, agent.state))
                S.update_actual_duration(agent.task_sequence[0], 1)
            else:
                pass
    
    S.append_number_of_collisions(Rs.detect_collisions())
    
    # Update Statistics for the total paths taken
    S.add_paths(Rs.get_agent_states())
    
    # Update Statistics for Asile and Driveway Occupancy
    S.add_aisle_occupancy(G.get_aisle_occupancy())
    S.add_driveway_occupancy(G.get_driveway_occupancy())
    # S.add_aisle_occupancy(G.get_aisle_occupied())
    # S.add_driveway_occupancy(G.get_driveway_occupied())
    
    # Update free_agents, to_pickup, and to_delivery
    # TODO: Update warehouse and driveway inventory when a robot has reached goal location (i.e. for both for loops below) 
          
    for agent in Rs.agents:
        if agent.status == 1:
            if agent.state == get_task_start_location(J, agent.task_sequence[0]):
                S.add_completed_to_pickup_task_id(agent.task_sequence[0])
                agent.status = 2
        elif agent.status == 2:
            if agent.state == get_task_goal_location(J, agent.task_sequence[0]):
                S.add_completed_task_id(agent.task_sequence[0])
                
                actual_duration = S.get_actual_duration(agent.task_sequence[0]) + S.get_actual_pickup_duration(agent.task_sequence[0])
                estimated_duration = S.get_estimated_duration(agent.task_sequence[0]) + S.get_estimated_pickup_duration(agent.task_sequence[0])
                    
                if np.abs((actual_duration - estimated_duration)/estimated_duration) > 0.1:
                    B.dump(agent.id, agent.task_sequence[0], J, S)
                
                for task in J:
                    if task[0] == agent.task_sequence[0]:
                        J.remove(task)
                        break
                    
                agent.task_sequence.pop(0)
                if agent.task_sequence == []:
                    agent.status = 0
                else:
                    agent.status = 1
                    S.add_actual_distance(agent.task_sequence[0])
                    S.add_actual_pickup_distance(agent.task_sequence[0])
                    
                    S.add_actual_duration(agent.task_sequence[0])
                    S.add_actual_pickup_duration(agent.task_sequence[0])
                    
                    estimated_to_pickup_path = dc.distance(map_name, agent.state, get_task_start_location(J, agent.task_sequence[0]))
                    estimated_task_path = dc.distance(map_name, get_task_start_location(J, agent.task_sequence[0]), get_task_goal_location(J, agent.task_sequence[0]))

                    S.add_estimated_pickup_duration(agent.task_sequence[0], estimated_to_pickup_path)
                    S.add_estimated_pickup_distance(agent.task_sequence[0], estimated_to_pickup_path)

                    S.add_estimated_distance(agent.task_sequence[0], estimated_task_path)
                    S.add_estimated_duration(agent.task_sequence[0], estimated_task_path)
                    
                    if t <= 100:
                        S.append_early_task_ids(agent.task_sequence[0])

    return Rs, J