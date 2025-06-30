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
                S.update_actual_pickup_distance(agent.task_sequence[0][0], euclidian_distance(old_state, agent.state))
                S.update_actual_pickup_duration(agent.task_sequence[0][0], S.get_actual_pickup_duration(agent.task_sequence[0][0]) + 1)
                
            elif agent.status == 2:
                S.update_actual_distance(agent.task_sequence[0][0], euclidian_distance(old_state, agent.state))
                S.update_actual_duration(agent.task_sequence[0][0], S.get_actual_duration(agent.task_sequence[0][0]) + 1)
            else:
                pass
    
    S.append_number_of_collisions(Rs.detect_collisions())
    
    # Update Statistics for the total paths taken
    S.add_paths(Rs.get_agent_states())
    
    # Update Statistics for Asile and Driveway Occupancy
    S.add_aisle_occupancy(G.get_aisle_occupancy())
    S.add_driveway_occupancy(G.get_driveway_occupancy())
    
    for agent in Rs.agents:
        if agent.status == 1:
            if agent.state == agent.task_sequence[0][1]:
                task = agent.task_sequence[0]
                task_id = task[0]
                start_location = task[1]
                goal_location = task[2]
                # Outbound: picking up from warehouse
                if start_location in G.warehouse.get_full_locations():
                    try:
                        G.warehouse.remove_sku_instance(start_location)
                    except Exception as e:
                        print(f"[WARN] Could not remove SKU from warehouse at {start_location}: {e}")
                # Inbound: picking up from driveway (now empty)
                elif start_location in G.driveway.get_full_locations():
                    try:
                        G.driveway.remove_sku_instance(start_location)
                    except Exception as e:
                        print(f"[WARN] Could not remove SKU from driveway at {start_location}: {e}")
                S.add_completed_to_pickup_task_id(task_id)
                S.update_actual_pickup_duration(task_id, t - S.get_actual_pickup_duration(task_id))
                agent.status = 2
        elif agent.status == 2:
            if agent.state == agent.task_sequence[0][2]:
                task = agent.task_sequence[0]
                task_id = task[0]
                start_location = task[1]
                goal_location = task[2]
                # Inbound: dropping off in warehouse
                # Try to get SKU ID if present in task tuple
                sku_id = None
                if len(task) > 3:
                    sku_id = task[3]
                if sku_id is not None:
                    try:
                        G.warehouse.add_sku_instance(sku_id, goal_location)
                    except Exception as e:
                        print(f"[WARN] Could not add SKU {sku_id} to warehouse at {goal_location}: {e}")
                S.add_completed_task_id(task_id, t, start_location, goal_location)
                S.update_service_time(task_id, t)
                
                for task in J:
                    if task[0] == task_id:
                        J.remove(task)
                        break
                    
                agent.task_sequence.pop(0)
                if agent.task_sequence == []:
                    agent.status = 0
                else:
                    agent.status = 1
                    new_task_id = agent.task_sequence[0][0]
                    S.add_actual_distance(new_task_id)
                    S.add_actual_pickup_distance(new_task_id)
                    
                    # Initialize durations for new task 
                    S.add_actual_duration(new_task_id)
                    S.add_actual_pickup_duration(new_task_id)
                    
                    estimated_to_pickup_path = dc.distance(map_name, agent.state, agent.task_sequence[0][1])
                    estimated_task_path = dc.distance(map_name, agent.task_sequence[0][1], agent.task_sequence[0][2])

                    S.add_estimated_pickup_duration(new_task_id, estimated_to_pickup_path)
                    S.add_estimated_pickup_distance(new_task_id, estimated_to_pickup_path)

                    S.add_estimated_distance(new_task_id, estimated_task_path)
                    S.add_estimated_duration(new_task_id, estimated_task_path)
                    
                    if t <= 100:
                        S.append_early_task_ids(new_task_id)

    return Rs, J