from typing import Tuple, Dict

from .analysis.statistics import Stats
from .graph import Graph
from .agent import *
from .utils import *

def simulate(S : Stats, G : Graph, Rs : AgentLoader, J : Dict[int, Tuple], map_name : str, t : int) -> Tuple[AgentLoader, set]:
    """
    Simulate the system for one timestep.

    Args:
        S (Stats): Statistics object
        G (Graph): Graph object
        Rs (AgentLoader): AgentLoader object
        J (Dict[int, Tuple]): Dictionary of tasks
        map_name (str): Name of the map
        t (int): Current timestep

    Returns:
        Tuple[AgentLoader, Dict[int, Tuple]]: Updated AgentLoader object and updated dictionary of tasks
    """

    # Update state of robots
    for agent in Rs.agents:
        # If robot sequence is stationary, leave the robot in place (wait action)
        if len(agent.path_sequence) == 0:
            continue
        # If robot does have a sequence of actions, pop next state and update
        else:
            # Update Agent State and Graph Occupied States
            G.set_occupied(agent.state, False)
            agent.state = agent.path_sequence.pop(0)
            G.set_occupied(agent.state, True)
            
            if agent.status == 1:
                # S.update_actual_pickup_distance(agent.task_sequence[0][0], euclidian_distance(old_state, agent.state))
                S.update_actual_pickup_duration(agent.task_sequence[0][0], S.get_actual_pickup_duration(agent.task_sequence[0][0]) + 1)
                
            elif agent.status == 2:
                # S.update_actual_distance(agent.task_sequence[0][0], euclidian_distance(old_state, agent.state))
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
                deadline = task[3]
                # Outbound: picking up from warehouse
                if start_location in G.warehouse.get_full_locations():
                    try:
                        sku_id = G.warehouse.get_sku_at_location(start_location).sku_id
                        agent.set_sku_id_carrying(G.warehouse.get_sku_at_location(start_location).sku_id)
                        G.warehouse.remove_sku_instance(start_location)
                        G.update_sku_KD_trees(agent.get_sku_id_carrying())
                        
                        # Update outbound tasks in J with the same sku_id to remove this location as a start location
                        for other_task_id in J.keys():
                            if other_task_id != task_id and J[other_task_id][3] == sku_id and J[other_task_id][4] == 0:
                                if start_location in J[other_task_id][0]:
                                    # Create new tuple with updated start locations
                                    new_start_locations = frozenset(G.warehouse.get_sku_instances(sku_id))
                                    new_task_tuple = (
                                        new_start_locations,
                                        J[other_task_id][1],
                                        J[other_task_id][2],
                                        J[other_task_id][3],
                                        J[other_task_id][4]
                                    )
                                    J[other_task_id] = new_task_tuple
                            # Also update inbound tasks to add this location to the goal locations regardless of sku_id
                            if other_task_id != task_id and J[other_task_id][4] == 1:
                                # Refresh inbound goal locations from current warehouse empties
                                new_goal_locations = frozenset(G.warehouse.get_empty_locations())
                                new_task_tuple = (
                                    J[other_task_id][0],
                                    new_goal_locations,
                                    J[other_task_id][2],
                                    J[other_task_id][3],
                                    J[other_task_id][4]
                                )
                                J[other_task_id] = new_task_tuple
                    except Exception as e:
                        print(f"[WARN] Could not remove SKU from warehouse at {start_location}: {e}")
                        exit()
                # Inbound: picking up from driveway (now empty)
                elif start_location in G.driveway.get_full_locations():
                    try:
                        # Only save sku id
                        agent.set_sku_id_carrying(G.driveway.get_sku_at_location(start_location).sku_id)
                        G.driveway.remove_sku_instance(start_location)
                        
                        # Refresh goal locations of all outbound tasks to current driveway empties
                        for other_task_id in J.keys():
                            if other_task_id != task_id and J[other_task_id][4] == 0:
                                new_goal_locations = frozenset(G.driveway.get_empty_locations())
                                new_task_tuple = (
                                    J[other_task_id][0],
                                    new_goal_locations,
                                    J[other_task_id][2],
                                    J[other_task_id][3],
                                    J[other_task_id][4]
                                )
                                J[other_task_id] = new_task_tuple
                    except Exception as e:
                        print(f"[WARN] Could not remove SKU from driveway at {start_location}: {e}")

                S.add_completed_to_pickup_task_id(task_id)
                agent.status = 2
        elif agent.status == 2:
            if agent.state == agent.task_sequence[0][2]:
                task = agent.task_sequence[0]
                task_id = task[0]
                start_location = task[1]
                goal_location = task[2]
                

                deadline = J[task_id][2]
                sku_id = J[task_id][3]
                inbound_task = J[task_id][4]

                # Inbound task: dropping off to warehouse
                if goal_location in G.warehouse.get_empty_locations():
                    G.warehouse.add_sku_instance(agent.get_sku_id_carrying(), goal_location)
                    G.update_sku_KD_trees(agent.get_sku_id_carrying())
                    # Update inbound tasks in J to remove this location as a goal location for all sku_ids
                    for other_task_id in J.keys():
                        if other_task_id != task_id and J[other_task_id][4] == 1:
                            if goal_location in J[other_task_id][1]:
                                # Create new tuple with updated goal locations
                                new_goal_locations = frozenset(G.warehouse.get_empty_locations())
                                new_task_tuple = (
                                    J[other_task_id][0],
                                    new_goal_locations,
                                    J[other_task_id][2],
                                    J[other_task_id][3],
                                    J[other_task_id][4]
                                )
                                J[other_task_id] = new_task_tuple
                        # Update outbound tasks to add this location to the start locations if the sku_id matches
                        if other_task_id != task_id and J[other_task_id][4] == 0 and J[other_task_id][3] == sku_id:
                            if goal_location not in J[other_task_id][0]:
                                # Create new tuple with updated start locations
                                new_start_locations = frozenset(G.warehouse.get_sku_instances(sku_id))
                                new_task_tuple = (
                                    new_start_locations,
                                    J[other_task_id][1],
                                    J[other_task_id][2],
                                    J[other_task_id][3],
                                    J[other_task_id][4]
                                )
                                J[other_task_id] = new_task_tuple
                                
                elif goal_location in G.driveway.get_empty_locations():
                    pass
                agent.set_sku_id_carrying(None)

                S.add_completed_task_id(task_id, t, start_location, goal_location, int(deadline), int(sku_id), int(inbound_task))
                S.update_service_time(task_id, t)
                
                J.pop(task_id)
                    
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

                    estimated_to_pickup_path = G.get_distance(agent.state, agent.task_sequence[0][1])
                    estimated_task_path = G.get_distance(agent.task_sequence[0][1], agent.task_sequence[0][2])

                    S.add_estimated_pickup_duration(new_task_id, estimated_to_pickup_path)
                    S.add_estimated_pickup_distance(new_task_id, estimated_to_pickup_path)

                    S.add_estimated_distance(new_task_id, estimated_task_path)
                    S.add_estimated_duration(new_task_id, estimated_task_path)
                    
                    if t <= 100:
                        S.append_early_task_ids(new_task_id)

    return Rs, J